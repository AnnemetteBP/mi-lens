"""End-to-end capture and held-out analysis for RouterInterp-style SAEs."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from mi_lens.sparse import (
    RouterInterpCaptureConfig,
    ITDA,
    ITDAConfig,
    TopKSAEConfig,
    TopKSAE,
    capture_flexolmo_router_layers,
    capture_routerinterp_prompt_artifacts,
    feature_activation_diagnostics,
    feature_coactivation_ratio,
    fit_routing_probe,
    fit_topk_sae,
    macro_f1_expert_routing,
    rho_usefulness,
    routing_recall,
    selected_expert_targets,
    sparse_by_magnitude,
    top_k_expert_predictions,
    top_rho_features,
)
from mi_lens.adapters.flex_olmo import iter_flex_olmo_layers
from mi_lens.methods.router_data_prep import iter_router_records_from_config

from .common import (
    ModelLoadConfig,
    clear_runtime_state,
    load_jsonl_records,
    load_model_and_tokenizer,
    metadata_payload,
    slugify,
    write_json,
)


def _require_finite_tensor(name: str, values: torch.Tensor) -> None:
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinity.")


def _require_probability_matrix(name: str, values: torch.Tensor) -> None:
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError(f"{name} must be shaped (token, expert).")
    _require_finite_tensor(name, values)
    tolerance = 2e-3 if values.dtype in (torch.float16, torch.bfloat16) else 1e-5
    if (values < -tolerance).any() or (values > 1.0 + tolerance).any():
        raise ValueError(f"{name} contains values outside [0, 1].")
    sums = values.float().sum(-1)
    if not torch.allclose(sums, torch.ones_like(sums), rtol=tolerance, atol=tolerance):
        raise ValueError(f"{name} rows are not normalised.")


def _checked_metric(name: str, value: float, *, lower: float | None = None, upper: float | None = None) -> float:
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError(f"{name} is NaN or infinity.")
    tolerance = 1e-5
    if lower is not None and value < lower - tolerance:
        raise ValueError(f"{name}={value} is below {lower}.")
    if upper is not None and value > upper + tolerance:
        raise ValueError(f"{name}={value} is above {upper}.")
    return float(value)


def _require_json_finite(value: Any, *, path: str = "result") -> None:
    """Prevent JSON summaries from serialising a latent NaN/Inf anywhere."""

    if isinstance(value, dict):
        for key, nested in value.items():
            _require_json_finite(nested, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _require_json_finite(nested, path=f"{path}[{index}]")
    elif isinstance(value, float) and not torch.isfinite(torch.tensor(value)):
        raise ValueError(f"{path} is NaN or infinity.")


def _project_tmp_path(project_root: Path, relative_path: str) -> Path:
    """Keep RouterInterp artifacts on the project volume, never a home directory."""

    configured = Path(relative_path)
    if configured.is_absolute():
        raise ValueError("RouterInterp output paths must be relative to the project tmp directory.")
    tmp_root = (project_root / "tmp").resolve()
    destination = (project_root / configured).resolve()
    if destination != tmp_root and tmp_root not in destination.parents:
        raise ValueError("RouterInterp output paths must be located under project_root/tmp/.")
    return destination


def _routerinterp_output_dir(config: dict[str, Any]) -> Path:
    project_root = Path(config["project_root"]).resolve()
    if "output_path" in config:
        return _project_tmp_path(project_root, str(config["output_path"]))
    family = str(config.get("model_family", "flexolmo"))
    label = str(config.get("model_label") or slugify(config["model"]["model_name"]))
    split = str(config.get("dataset_split_role", "train"))
    return project_root / "tmp" / "routerinterp" / family / label / split


def _iter_jsonl_records(paths: list[Path]):
    """Yield pooled source-order records without holding a corpus in memory."""

    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def run_routerinterp_streaming_sae_fit_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Fit RouterInterp SAEs directly from model activations without an activation cache."""

    project_root = Path(config["project_root"]).resolve()
    output_dir = _project_tmp_path(project_root, str(config["output_path"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    if "dataset_config_path" in config:
        paths: list[Path] = []
        records = iter_router_records_from_config(
            project_root / str(config["dataset_config_path"]),
            project_root=project_root,
            role=str(config.get("dataset_role", "sae_fit")),
        )
    else:
        paths = sorted(project_root.glob(str(config["examples_glob"])))
        if not paths:
            raise FileNotFoundError(f"No JSONL files matched {config['examples_glob']!r}.")
        records = _iter_jsonl_records(paths)
    model_config = ModelLoadConfig(**config["model"])
    model, tokenizer, cache_dir = load_model_and_tokenizer(model_config, project_root=project_root)
    model.eval()
    device = next(model.parameters()).device
    target_tokens = int(config["max_sae_fit_tokens"])
    batch_size = int(config.get("sae", {}).get("batch_size", 1024))
    learning_rate = float(config.get("sae", {}).get("learning_rate", 3e-4))
    if target_tokens < 1 or batch_size < 1:
        raise ValueError("max_sae_fit_tokens and sae.batch_size must be positive.")
    try:
        requested_layers = config["layers"]
        model_layers = list(iter_flex_olmo_layers(model))
        if requested_layers == "routerinterp_quartiles":
            layer_count = len(model_layers)
            layers = tuple((index + 1) * layer_count // 4 - 1 for index in range(4))
        else:
            layers = tuple(int(layer) for layer in requested_layers)
        expert_labels = tuple(str(value) for value in config.get("expert_labels", ()))
        max_seq_len = config.get("max_seq_len")
        buffers: dict[int, torch.Tensor] = {}
        saes: dict[int, TopKSAE] = {}
        optimizers: dict[int, torch.optim.Optimizer] = {}
        losses: dict[int, list[float]] = {layer: [] for layer in layers}
        itda_settings = dict(config.get("itda", {}))
        itda_enabled = bool(itda_settings.get("enabled", False))
        itda_fit_tokens = int(itda_settings.get("fit_tokens", 1_000_000)) if itda_enabled else 0
        itda_batch_size = int(itda_settings.get("batch_size", 64)) if itda_enabled else 0
        if itda_enabled and (itda_fit_tokens < 1 or itda_batch_size < 1):
            raise ValueError("ITDA fit_tokens and batch_size must be positive.")
        # ITDA is documented as a lightweight, million-token alternative to
        # SAE training.  Select a fixed source-order stride from the exact SAE
        # stream so no second model pass or activation cache is required.
        # Floor rather than ceil so the deterministic source-order sample has
        # at least the requested budget before the final exact truncation.
        itda_stride = max(1, target_tokens // itda_fit_tokens) if itda_enabled else 0
        itdas: dict[int, ITDA] = {}
        itda_history: dict[int, list[dict[str, float | int]]] = {layer: [] for layer in layers}
        itda_sampled_tokens = 0
        dataset_tokens: dict[str, int] = {}
        trained_tokens = 0

        for record in records:
            if trained_tokens >= target_tokens:
                break
            prompt = record.get("prompt", record.get("text"))
            if prompt is None:
                continue
            tokenized = tokenizer(
                str(prompt), return_tensors="pt", truncation=max_seq_len is not None,
                **({"max_length": int(max_seq_len)} if max_seq_len is not None else {}),
            )
            input_ids = tokenized.input_ids.to(device)
            captures = capture_flexolmo_router_layers(model, {"input_ids": input_ids})
            remaining = target_tokens - trained_tokens
            token_count = min(int(input_ids.shape[1]), remaining)
            if token_count < 1:
                continue
            dataset_name = str(record.get("dataset_name", record.get("task", "unknown")))
            dataset_tokens[dataset_name] = dataset_tokens.get(dataset_name, 0) + token_count
            sample_indices = None
            if itda_enabled and itda_sampled_tokens < itda_fit_tokens:
                offsets = torch.arange(token_count, device=device) + trained_tokens
                sample_indices = torch.nonzero(offsets.remainder(itda_stride) == 0, as_tuple=False).flatten()
                sample_indices = sample_indices[: itda_fit_tokens - itda_sampled_tokens]
            for layer in layers:
                capture = captures[layer]
                width = int(capture.router_probabilities.shape[-1])
                if expert_labels and len(expert_labels) != width:
                    raise ValueError(f"Configured {len(expert_labels)} labels for a {width}-expert router.")
                x = capture.router_input[0, :token_count].detach()
                if layer not in saes:
                    protocol = _resolve_routerinterp_protocol(config, d_model=int(x.shape[-1]))
                    sae_config = TopKSAEConfig(
                        d_model=int(x.shape[-1]),
                        n_features=int(protocol["sae"].get("n_features", x.shape[-1] * 4)),
                        k=int(protocol["sae"].get("k", 64)),
                    )
                    sae = TopKSAE(sae_config).to(device=device, dtype=x.dtype)
                    saes[layer] = sae
                    optimizers[layer] = torch.optim.Adam(sae.parameters(), lr=learning_rate)
                    buffers[layer] = x.new_empty((0, x.shape[-1]))
                buffers[layer] = torch.cat((buffers[layer], x), dim=0)
                while buffers[layer].shape[0] >= batch_size:
                    batch, buffers[layer] = buffers[layer][:batch_size], buffers[layer][batch_size:]
                    reconstruction, _ = saes[layer](batch)
                    loss = torch.nn.functional.mse_loss(reconstruction, batch)
                    optimizers[layer].zero_grad(set_to_none=True)
                    loss.backward()
                    optimizers[layer].step()
                    saes[layer].normalize_decoder()
                    losses[layer].append(float(loss.detach().item()))
                if itda_enabled and sample_indices is not None and sample_indices.numel():
                    if layer not in itdas:
                        itda_config = ITDAConfig(
                            d_model=int(x.shape[-1]),
                            max_atoms=int(itda_settings.get("max_atoms", x.shape[-1] * 16)),
                            k=int(itda_settings.get("k", 32)),
                            loss_threshold=float(itda_settings.get("loss_threshold", 0.01)),
                        )
                        itdas[layer] = ITDA(itda_config).to(device=device, dtype=x.dtype)
                    samples = x.index_select(0, sample_indices)
                    # Retain globally stable source positions for atom provenance.
                    sample_provenance = torch.stack(
                        (
                            sample_indices + trained_tokens,
                            sample_indices,
                        ),
                        dim=1,
                    )
                    for start in range(0, samples.shape[0], itda_batch_size):
                        itda_history[layer].append(
                            itdas[layer].update(
                                samples[start : start + itda_batch_size],
                                source_indices=sample_provenance[start : start + itda_batch_size],
                            )
                        )
            if sample_indices is not None:
                itda_sampled_tokens += int(sample_indices.numel())
            trained_tokens += token_count

        if trained_tokens != target_tokens:
            raise ValueError(f"Only {trained_tokens:,} source tokens available; need {target_tokens:,} for SAE fitting.")
        if itda_enabled and itda_sampled_tokens != itda_fit_tokens:
            raise ValueError(
                f"ITDA received {itda_sampled_tokens:,} source tokens; expected exactly {itda_fit_tokens:,}."
            )
        for layer, sae in saes.items():
            # Train on any final short batch so the declared token budget is exact.
            if buffers[layer].numel():
                reconstruction, _ = sae(buffers[layer])
                loss = torch.nn.functional.mse_loss(reconstruction, buffers[layer])
                optimizers[layer].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[layer].step()
                sae.normalize_decoder()
                losses[layer].append(float(loss.detach().item()))
            layer_dir = output_dir / f"layer_{layer:02d}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            sae.eval().save_pretrained(str(layer_dir / "topk_sae.pt"))
            if itda_enabled:
                if layer not in itdas:
                    raise ValueError(f"ITDA did not receive any sampled router activations at layer {layer}.")
                itdas[layer].eval().save_pretrained(str(layer_dir / "itda.pt"))
            gate_weight = model_layers[layer].mlp.gate.weight.detach().cpu().float()
            if gate_weight.ndim != 2 or gate_weight.shape[1] != sae.config.d_model:
                raise ValueError(f"Layer {layer} gate weights are incompatible with its SAE input width.")
            torch.save({"router_weight": gate_weight}, layer_dir / "router_geometry.pt")
        manifest = {
            "format": "mi_lens.routerinterp.streaming_sae.v1",
            "trained_tokens_per_layer": trained_tokens,
            "layers": list(layers),
            "dataset_tokens": dataset_tokens,
            "expert_labels": list(expert_labels),
            "source_paths": [str(path) for path in paths],
            "dataset_config_path": config.get("dataset_config_path"),
            "final_loss": {str(layer): values[-1] for layer, values in losses.items() if values},
            "itda": {
                "enabled": itda_enabled,
                "sampled_tokens_per_layer": itda_sampled_tokens if itda_enabled else 0,
                "source_stride": itda_stride if itda_enabled else None,
                "source_order_sampling": "global token index modulo source_stride" if itda_enabled else None,
                "config": itda_settings if itda_enabled else None,
                "final_update": {
                    str(layer): values[-1] for layer, values in itda_history.items() if values
                },
            },
            "cache_dir": str(cache_dir),
        }
        write_json(output_dir / "sae_fit_manifest.json", manifest)
        return {"output_dir": str(output_dir), "manifest_path": str(output_dir / "sae_fit_manifest.json"), **manifest}
    finally:
        clear_runtime_state(model, tokenizer)


def run_routerinterp_capture_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Capture pre-router vectors and routing labels from a fixed JSONL split."""

    project_root = Path(config["project_root"]).resolve()
    model_config = ModelLoadConfig(**config["model"])
    output_dir = _routerinterp_output_dir(config)
    if "dataset_config_path" in config:
        example_paths: list[Path] = []
        records = list(iter_router_records_from_config(
            project_root / str(config["dataset_config_path"]),
            project_root=project_root,
            role=str(config.get("dataset_role", "eval")),
        ))
    else:
        if "examples_path" in config:
            example_paths = [project_root / str(config["examples_path"])]
        elif "examples_glob" in config:
            example_paths = sorted(project_root.glob(str(config["examples_glob"])))
            if not example_paths:
                raise FileNotFoundError(f"No JSONL files matched {config['examples_glob']!r}.")
        else:
            raise KeyError("RouterInterp capture requires a dataset config, examples_path, or examples_glob.")
        records = []
        for example_path in example_paths:
            records.extend(load_jsonl_records(example_path))
    if config.get("max_examples") is not None:
        records = records[: int(config["max_examples"])]
    model, tokenizer, cache_dir = load_model_and_tokenizer(model_config, project_root=project_root)
    tokenizer_provenance = {
        "tokenizer_class_loaded": type(tokenizer).__name__,
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "tokenizer_vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
    }
    model.eval()
    try:
        requested_layers = config["layers"]
        if requested_layers == "routerinterp_quartiles":
            layer_count = len(iter_flex_olmo_layers(model))
            if layer_count < 4:
                raise ValueError("RouterInterp quartile selection requires at least four layers.")
            layers = tuple((index + 1) * layer_count // 4 - 1 for index in range(4))
        else:
            layers = tuple(int(layer) for layer in requested_layers)
        capture_config = RouterInterpCaptureConfig(
            layers=layers,
            expert_labels=tuple(str(label) for label in config.get("expert_labels", ())),
            max_seq_len=None if config.get("max_seq_len") is None else int(config["max_seq_len"]),
            skip_first_token=bool(config.get("skip_first_token", False)),
            artifact_dtype=str(config.get("artifact_dtype", "bfloat16")),
            max_tokens=None if config.get("max_tokens") is None else int(config["max_tokens"]),
        )
        manifest = capture_routerinterp_prompt_artifacts(
            model,
            tokenizer,
            records,
            output_dir=output_dir,
            config=capture_config,
        )
    finally:
        clear_runtime_state(model, tokenizer)

    metadata_path = output_dir / "capture_metadata.json"
    write_json(
        metadata_path,
        {
            "metadata": metadata_payload(
                model_config=model_config,
                cache_dir=cache_dir,
                extra={
                    "output_dir": str(output_dir),
                    "examples_paths": [str(path) for path in example_paths],
                    "dataset_split_role": config.get("dataset_split_role", "train"),
                    "layers": list(capture_config.layers),
                    **tokenizer_provenance,
                },
            ),
            "manifest_path": str(output_dir / "manifest.json"),
        },
    )
    return {
        "output_dir": str(output_dir),
        "manifest_path": str(output_dir / "manifest.json"),
        "metadata_path": str(metadata_path),
        "num_prompts": manifest["num_prompts"],
        "num_tokens": manifest["num_tokens"],
        "layers": list(capture_config.layers),
    }


def _load_layer_tokens(
    artifact_dir: Path,
    *,
    layer: int,
    max_tokens: int | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
    list[str],
    list[str],
    int,
    int,
]:
    """Read an explicit token cap from per-prompt artifacts in manifest order."""

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    activations: list[torch.Tensor] = []
    selected: list[torch.Tensor] = []
    router_probabilities: list[torch.Tensor] = []
    token_ids: list[torch.Tensor] = []
    previous_token_ids: list[torch.Tensor] = []
    domains: list[str] = []
    languages: list[str] = []
    datasets: list[str] = []
    retained = 0
    num_experts: int | None = None
    for row in manifest["prompts"]:
        payload = torch.load(artifact_dir / row["path"], map_location="cpu", weights_only=True)
        try:
            block = payload["layers"][str(layer)]
        except KeyError as exc:
            raise KeyError(f"Layer {layer} is absent from {artifact_dir / row['path']}") from exc
        x = block["router_input"].float()
        y = block["selected_experts"].long()
        probabilities = block["router_probabilities"].float()
        positions = payload["positions"].long()
        full_token_ids = payload["token_ids"].long()
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError(f"Malformed RouterInterp artifact {artifact_dir / row['path']}.")
        if probabilities.ndim != 2 or probabilities.shape[0] != x.shape[0]:
            raise ValueError(f"Malformed router probabilities in {artifact_dir / row['path']}.")
        _require_finite_tensor(f"router inputs in {row['path']}", x)
        _require_probability_matrix(f"router probabilities in {row['path']}", probabilities)
        if y.shape[1] < 1 or (y < 0).any() or (y >= probabilities.shape[1]).any():
            raise ValueError(f"Malformed selected expert ids in {artifact_dir / row['path']}.")
        if (y.sort(dim=1).values[:, 1:] == y.sort(dim=1).values[:, :-1]).any():
            raise ValueError(f"Duplicate selected experts in {artifact_dir / row['path']}.")
        if positions.ndim != 1 or positions.shape[0] != x.shape[0]:
            raise ValueError(f"Malformed token positions in {artifact_dir / row['path']}.")
        if full_token_ids.ndim != 1 or positions.max().item() >= full_token_ids.shape[0]:
            raise ValueError(f"Malformed source token ids in {artifact_dir / row['path']}.")
        source_ids = full_token_ids[positions]
        previous_ids = torch.where(positions > 0, full_token_ids[(positions - 1).clamp_min(0)], -1)
        width = int(probabilities.shape[1])
        if num_experts is not None and num_experts != width:
            raise ValueError(f"Inconsistent expert width in {artifact_dir} layer {layer}.")
        num_experts = width
        if max_tokens is not None:
            remaining = max_tokens - retained
            if remaining <= 0:
                break
            x, y, probabilities = x[:remaining], y[:remaining], probabilities[:remaining]
            source_ids, previous_ids = source_ids[:remaining], previous_ids[:remaining]
        activations.append(x)
        selected.append(y)
        router_probabilities.append(probabilities)
        token_ids.append(source_ids)
        previous_token_ids.append(previous_ids)
        metadata = {
            key: str(payload.get(key, "")).strip()
            for key in ("domain", "language", "dataset_name")
        }
        missing = [key for key, value in metadata.items() if not value or value.lower() == "unknown"]
        if missing:
            raise ValueError(
                f"RouterInterp artifact {artifact_dir / row['path']} is missing required metadata: "
                + ", ".join(missing)
            )
        domains.extend([metadata["domain"]] * x.shape[0])
        languages.extend([metadata["language"]] * x.shape[0])
        datasets.extend([metadata["dataset_name"]] * x.shape[0])
        retained += x.shape[0]
    if not activations:
        raise ValueError(f"No tokens available for layer {layer} in {artifact_dir}.")
    assert num_experts is not None
    return (
        torch.cat(activations),
        torch.cat(selected),
        torch.cat(router_probabilities),
        torch.cat(token_ids),
        torch.cat(previous_token_ids),
        domains,
        languages,
        datasets,
        retained,
        num_experts,
    )


def _load_token_provenance(
    artifact_dir: Path,
    *,
    max_tokens: int | None,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    """Keep lightweight prompt provenance aligned with flattened token rows.

    Activations remain in the regular tensors.  This helper stores only the
    information needed to inspect the handful of held-out contexts selected by
    a top-rho feature later in the analysis.
    """

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    prompts: list[dict[str, Any]] = []
    prompt_rows: list[torch.Tensor] = []
    token_positions: list[torch.Tensor] = []
    retained = 0
    for row in manifest["prompts"]:
        if max_tokens is not None and retained >= max_tokens:
            break
        payload = torch.load(artifact_dir / row["path"], map_location="cpu", weights_only=True)
        positions = payload["positions"].long()
        if max_tokens is not None:
            positions = positions[: max_tokens - retained]
        if not len(positions):
            continue
        prompt_index = len(prompts)
        prompts.append(
            {
                "example_id": str(payload.get("example_id", row.get("example_id", prompt_index))),
                "dataset_name": str(payload.get("dataset_name", row.get("dataset_name", "unknown"))),
                "task": str(payload.get("task", row.get("task", "unknown"))),
                "domain": str(payload.get("domain", row.get("domain", "unknown"))),
                "language": str(payload.get("language", row.get("language", "unknown"))),
                "prompt": str(payload.get("prompt", "")),
                "token_ids": payload["token_ids"].long(),
            }
        )
        prompt_rows.append(torch.full((len(positions),), prompt_index, dtype=torch.long))
        token_positions.append(positions)
        retained += len(positions)
    if not prompt_rows:
        raise ValueError(f"No token provenance found in {artifact_dir}.")
    return prompts, torch.cat(prompt_rows), torch.cat(token_positions)


def _top_rho_feature_contexts(
    *,
    features: torch.Tensor,
    rho_indices: torch.Tensor,
    rho_values: torch.Tensor,
    expert_labels: list[str],
    prompt_records: list[dict[str, Any]],
    prompt_rows: torch.Tensor,
    token_positions: torch.Tensor,
    max_contexts_per_feature: int,
) -> list[dict[str, Any]]:
    """Save actual held-out contexts for the most routing-useful SAE latents."""

    if features.shape[0] != prompt_rows.numel() or features.shape[0] != token_positions.numel():
        raise ValueError("Feature activations and token provenance are not aligned.")
    if max_contexts_per_feature < 1:
        raise ValueError("max_contexts_per_feature must be positive.")
    _require_finite_tensor("feature-context activations", features)
    _require_finite_tensor("rho values", rho_values)
    rows: list[dict[str, Any]] = []
    for expert in range(rho_indices.shape[0]):
        for rank, feature_id_tensor in enumerate(rho_indices[expert]):
            feature_id = int(feature_id_tensor.item())
            # ITDA coefficients can be signed. Domain concentration concerns
            # feature use, so measure activation magnitude for both methods.
            signed_activations = features[:, feature_id]
            activations = signed_activations.abs()
            count = min(max_contexts_per_feature, activations.numel())
            values, indices = activations.topk(k=count)
            for context_rank, (value, token_row) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
                prompt = prompt_records[int(prompt_rows[token_row].item())]
                token_position = int(token_positions[token_row].item())
                token_ids = prompt["token_ids"]
                rows.append(
                    {
                        "expert": expert,
                        "expert_label": expert_labels[expert] if expert_labels else f"expert_{expert}",
                        "feature_id": feature_id,
                        "rho_rank": rank + 1,
                        "rho": float(rho_values[expert, rank].item()),
                        "context_rank": context_rank,
                        "activation": float(signed_activations[token_row].item()),
                        "activation_magnitude": float(value),
                        "example_id": prompt["example_id"],
                        "dataset_name": prompt["dataset_name"],
                        "task": prompt["task"],
                        "domain": prompt["domain"],
                        "language": prompt["language"],
                        "token_position": token_position,
                        "token_id": int(token_ids[token_position].item()),
                        "previous_token_id": int(token_ids[token_position - 1].item()) if token_position else None,
                        # The source text is retained for manual feature interpretation;
                        # it is never presented as an automatic semantic explanation.
                        "prompt": prompt["prompt"],
                    }
                )
    return rows


def _top_rho_feature_domain_profiles(
    *,
    features: torch.Tensor,
    domains: list[str],
    rho_indices: torch.Tensor,
    rho_values: torch.Tensor,
    expert_labels: list[str],
) -> list[dict[str, Any]]:
    """Describe top-rho latent firing across observed dataset domains only."""

    if features.shape[0] != len(domains):
        raise ValueError("Feature activations and domain labels are not aligned.")
    _require_finite_tensor("feature-domain activations", features)
    _require_finite_tensor("feature-domain rho values", rho_values)
    domain_names = sorted(set(domains))
    rows: list[dict[str, Any]] = []
    for expert in range(rho_indices.shape[0]):
        for rank, feature_id_tensor in enumerate(rho_indices[expert]):
            feature_id = int(feature_id_tensor.item())
            # ITDA matching-pursuit coefficients can be negative; distribution
            # summaries therefore use the magnitude of feature use.
            activations = features[:, feature_id].abs()
            masses = []
            means: dict[str, float] = {}
            for domain in domain_names:
                mask = torch.tensor([value == domain for value in domains], device=features.device)
                domain_activations = activations[mask]
                masses.append(domain_activations.sum())
                means[domain] = float(domain_activations.mean().item()) if domain_activations.numel() else 0.0
            mass_tensor = torch.stack(masses)
            shares = mass_tensor / mass_tensor.sum().clamp_min(1e-8)
            _require_finite_tensor("feature-domain activation shares", shares)
            if (shares < -1e-8).any() or (shares > 1.0 + 1e-8).any():
                raise ValueError("Feature-domain activation shares are outside [0, 1].")
            positive = shares[shares > 0]
            entropy = (
                float((-(positive * positive.log()).sum() / math.log(len(domain_names))).item())
                if len(domain_names) > 1 and positive.numel()
                else 0.0
            )
            for domain_index, domain in enumerate(domain_names):
                rows.append(
                    {
                        "expert": expert,
                        "expert_label": expert_labels[expert] if expert_labels else f"expert_{expert}",
                        "feature_id": feature_id,
                        "rho_rank": rank + 1,
                        "rho": float(rho_values[expert, rank].item()),
                        "domain": domain,
                        "mean_activation_magnitude": means[domain],
                        "activation_mass_share": float(shares[domain_index].item()),
                        "normalized_domain_entropy": entropy,
                    }
                )
    return rows


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Return average ranks while handling ties without a SciPy dependency."""

    if values.ndim != 1:
        raise ValueError("Ranks require a one-dimensional tensor.")
    order = torch.argsort(values)
    sorted_values = values[order]
    ranks = torch.empty_like(values, dtype=torch.float32)
    start = 0
    while start < values.numel():
        end = start + 1
        while end < values.numel() and bool(sorted_values[end] == sorted_values[start]):
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _spearman_correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
    """Rank correlation, returning ``None`` for a constant vector."""

    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Spearman correlation requires equally shaped vectors.")
    left_ranks = _average_ranks(left.float())
    right_ranks = _average_ranks(right.float())
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = left_centered.norm() * right_centered.norm()
    if float(denominator.item()) == 0.0:
        return None
    return float((left_centered @ right_centered / denominator).item())


def _router_geometry_agreement(
    *,
    sae: TopKSAE,
    rho: torch.Tensor,
    router_weights: torch.Tensor,
    top_n: int,
) -> list[dict[str, float | int | list[int] | None]]:
    """Compare rho-based feature selection with router-weight geometry."""

    decoder_directions = sae.decoder.weight.detach().float().T
    if router_weights.ndim != 2 or router_weights.shape[1] != decoder_directions.shape[1]:
        raise ValueError("Router weights and SAE decoder directions have incompatible widths.")
    if router_weights.shape[0] != rho.shape[0]:
        raise ValueError("Router weight rows do not match the router expert count.")
    normalized_router = router_weights / router_weights.norm(dim=1, keepdim=True).clamp_min(1e-8)
    normalized_decoder = decoder_directions / decoder_directions.norm(dim=1, keepdim=True).clamp_min(1e-8)
    cosine_scores = normalized_router @ normalized_decoder.T
    output = []
    for expert in range(rho.shape[0]):
        rho_top = rho[expert].topk(k=min(top_n, rho.shape[1])).indices
        cosine_top = cosine_scores[expert].topk(k=min(top_n, cosine_scores.shape[1])).indices
        overlap = set(rho_top.cpu().tolist()) & set(cosine_top.cpu().tolist())
        union = set(rho_top.cpu().tolist()) | set(cosine_top.cpu().tolist())
        output.append(
            {
                "expert": expert,
                "spearman_rho_vs_router_cosine": _spearman_correlation(rho[expert], cosine_scores[expert]),
                "top_feature_overlap_count": len(overlap),
                "top_feature_jaccard": float(len(overlap) / len(union)) if union else 1.0,
                "rho_feature_ids": rho_top.cpu().tolist(),
                "router_cosine_feature_ids": cosine_top.cpu().tolist(),
            }
        )
    return output


def _domain_routing_summary(
    selected_experts: torch.Tensor,
    domains: list[str],
    *,
    num_experts: int,
) -> dict[str, Any]:
    """Compare each expert's dataset-domain mix with the corpus-wide mix."""

    if len(domains) != selected_experts.shape[0]:
        raise ValueError("Each routed token must retain one domain label.")
    names = sorted(set(domains))
    index = {name: value for value, name in enumerate(names)}
    domain_ids = torch.tensor([index[name] for name in domains], dtype=torch.long)
    token_counts = torch.bincount(domain_ids, minlength=len(names)).float()
    corpus_distribution = token_counts / token_counts.sum().clamp_min(1)

    def normalized_entropy(probabilities: torch.Tensor) -> float:
        if probabilities.numel() < 2:
            return 0.0
        positive = probabilities[probabilities > 0]
        return float((-(positive * positive.log()).sum() / torch.log(torch.tensor(float(probabilities.numel())))).item())

    per_expert = []
    for expert in range(num_experts):
        selected = (selected_experts == expert).any(dim=1)
        counts = torch.bincount(domain_ids[selected], minlength=len(names)).float()
        probabilities = counts / counts.sum().clamp_min(1)
        per_expert.append(
            {
                "expert": expert,
                "routed_token_count": int(selected.sum().item()),
                "normalized_domain_entropy": normalized_entropy(probabilities),
                "domain_routing_share": {name: float(probabilities[index[name]].item()) for name in names},
            }
        )
    result = {
        "domains": names,
        "corpus_normalized_entropy": normalized_entropy(corpus_distribution),
        "corpus_domain_share": {name: float(corpus_distribution[index[name]].item()) for name in names},
        "per_expert": per_expert,
    }
    _require_json_finite(result, path="domain_routing")
    return result


def _expert_activation_by_group(
    selected_experts: torch.Tensor,
    groups: list[str],
    *,
    num_experts: int,
) -> dict[str, Any]:
    """Report actual router selection rates conditional on a provenance group.

    This is intentionally the reverse conditional of ``_domain_routing_summary``:
    it answers ``P(expert active | Danish news)`` rather than merely describing
    which domains are present among tokens already routed to one expert.
    """

    targets = selected_expert_targets(selected_experts, num_experts)
    return _expert_activation_targets_by_group(targets, groups)


def _expert_activation_targets_by_group(
    targets: torch.Tensor,
    groups: list[str],
) -> dict[str, Any]:
    """Summarise binary expert activations conditional on provenance groups."""

    if targets.ndim != 2 or targets.shape[0] < 1:
        raise ValueError("Grouped expert activations must be a non-empty (token, expert) matrix.")
    if len(groups) != targets.shape[0]:
        raise ValueError("Each expert-activation target must retain one group label.")
    _require_finite_tensor("group expert activation targets", targets)
    if (targets < 0).any() or (targets > 1).any():
        raise ValueError("Grouped expert activations must be in [0, 1].")
    corpus_rates = targets.mean(dim=0)
    rows: list[dict[str, Any]] = []
    for group in sorted(set(groups)):
        mask = torch.tensor([value == group for value in groups], dtype=torch.bool)
        group_targets = targets[mask]
        rates = group_targets.mean(dim=0)
        _require_finite_tensor("group expert activation rates", rates)
        if (rates < 0).any() or (rates > 1).any():
            raise ValueError("Group expert activation rates are outside [0, 1].")
        rows.append(
            {
                "group": group,
                "token_count": int(mask.sum().item()),
                "expert_activation_rate": rates.tolist(),
                "expert_enrichment_vs_corpus": (rates / corpus_rates.clamp_min(1e-8)).tolist(),
            }
        )
    result = {
        "corpus_expert_activation_rate": corpus_rates.tolist(),
        "groups": rows,
    }
    _require_json_finite(result, path="expert_activation_by_group")
    return result


def _constant_top_k(selected_experts: torch.Tensor, split_name: str) -> int:
    if selected_experts.ndim != 2 or selected_experts.shape[1] < 1:
        raise ValueError(f"{split_name} selected-expert targets must be shaped (token, top_k).")
    return int(selected_experts.shape[1])


def _captured_token_count(artifact_dir: Path) -> int:
    """Read the manifest count before materialising any high-dimensional vectors."""

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    try:
        return sum(int(row["num_tokens"]) for row in manifest["prompts"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed RouterInterp manifest at {artifact_dir / 'manifest.json'}.") from exc


def _fit_ngram_counts(keys: torch.Tensor, targets: torch.Tensor) -> tuple[dict[tuple[int, ...], torch.Tensor], torch.Tensor]:
    """Count train-only token or token-pair co-occurrence with routed experts."""

    counts: dict[tuple[int, ...], torch.Tensor] = {}
    global_counts = targets.sum(0).float()
    for key, target in zip(keys.cpu().tolist(), targets.cpu()):
        normalized_key = tuple(key) if isinstance(key, list) else (int(key),)
        counts.setdefault(normalized_key, torch.zeros_like(global_counts)).add_(target.float())
    return counts, global_counts


def _predict_ngram_scores(
    keys: torch.Tensor,
    counts: dict[tuple[int, ...], torch.Tensor],
    global_counts: torch.Tensor,
) -> torch.Tensor:
    scores = []
    for key in keys.cpu().tolist():
        normalized_key = tuple(key) if isinstance(key, list) else (int(key),)
        scores.append(counts.get(normalized_key, global_counts))
    return torch.stack(scores)


def _expected_calibration_error(probabilities: torch.Tensor, targets: torch.Tensor, *, bins: int = 15) -> float:
    """Binary ECE for independent expert-selection probabilities."""

    if probabilities.shape != targets.shape:
        raise ValueError("Calibration probabilities and targets must have the same shape.")
    if bins < 2:
        raise ValueError("ECE requires at least two bins.")
    _require_finite_tensor("calibration probabilities", probabilities)
    _require_finite_tensor("calibration targets", targets)
    if (probabilities < 0).any() or (probabilities > 1).any() or (targets < 0).any() or (targets > 1).any():
        raise ValueError("Calibration probabilities and targets must be in [0, 1].")
    confidence = probabilities.flatten()
    outcome = targets.flatten().float()
    boundaries = torch.linspace(0, 1, bins + 1, device=confidence.device)
    error = torch.zeros((), device=confidence.device)
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        membership = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if membership.any():
            error += membership.float().mean() * (confidence[membership].mean() - outcome[membership].mean()).abs()
    return _checked_metric("selection_ece", float(error.item()), lower=0.0, upper=1.0)


def _router_distribution_metrics(scores: torch.Tensor, actual_probabilities: torch.Tensor) -> dict[str, float]:
    """Compare a score-derived expert distribution to captured router probabilities."""

    if scores.shape != actual_probabilities.shape:
        raise ValueError("Router scores and captured probabilities must have the same shape.")
    _require_finite_tensor("routing prediction scores", scores)
    _require_probability_matrix("captured router probabilities", actual_probabilities)
    if (scores >= 0).all():
        predicted = scores / scores.sum(-1, keepdim=True).clamp_min(1e-8)
    else:
        predicted = torch.softmax(scores, dim=-1)
    actual = actual_probabilities / actual_probabilities.sum(-1, keepdim=True).clamp_min(1e-8)
    predicted = predicted.clamp_min(1e-8)
    actual = actual.clamp_min(1e-8)
    midpoint = 0.5 * (actual + predicted)
    kl_actual_predicted = (actual * (actual.log() - predicted.log())).sum(-1).mean()
    jsd = 0.5 * (
        (actual * (actual.log() - midpoint.log())).sum(-1)
        + (predicted * (predicted.log() - midpoint.log())).sum(-1)
    ).mean()
    tv = 0.5 * (actual - predicted).abs().sum(-1).mean()
    return {
        "kl_actual_to_predicted": _checked_metric("kl_actual_to_predicted", float(kl_actual_predicted.item()), lower=0.0),
        "jensen_shannon_divergence": _checked_metric(
            "jensen_shannon_divergence", float(jsd.item()), lower=0.0, upper=float(torch.log(torch.tensor(2.0)).item())
        ),
        "total_variation_distance": _checked_metric("total_variation_distance", float(tv.item()), lower=0.0, upper=1.0),
    }


def _router_probability_distribution_summary(
    probabilities: torch.Tensor,
    *,
    bins: int = 24,
) -> dict[str, object]:
    """Compact, finite histogram summaries for appendix distribution figures."""

    if bins < 2:
        raise ValueError("Router distribution summaries require at least two bins.")
    _require_probability_matrix("captured router probabilities", probabilities)
    normalized = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-8)
    top_two = normalized.topk(k=min(2, normalized.shape[-1]), dim=-1).values
    top_one = top_two[:, 0]
    margin = top_one - top_two[:, 1] if normalized.shape[-1] > 1 else torch.zeros_like(top_one)
    entropy = -(normalized.clamp_min(1e-8) * normalized.clamp_min(1e-8).log()).sum(-1)
    entropy = entropy / math.log(normalized.shape[-1]) if normalized.shape[-1] > 1 else torch.zeros_like(entropy)

    def summarize(values: torch.Tensor, name: str) -> dict[str, object]:
        _require_finite_tensor(name, values)
        if (values < 0).any() or (values > 1).any():
            raise ValueError(f"{name} must be in [0, 1].")
        counts = torch.histc(values, bins=bins, min=0.0, max=1.0).to(torch.int64)
        edges = torch.linspace(0.0, 1.0, bins + 1)
        result = {
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()),
            "quantiles": {
                "p05": float(torch.quantile(values, 0.05).item()),
                "p25": float(torch.quantile(values, 0.25).item()),
                "p50": float(torch.quantile(values, 0.50).item()),
                "p75": float(torch.quantile(values, 0.75).item()),
                "p95": float(torch.quantile(values, 0.95).item()),
            },
            "histogram": {"edges": edges.tolist(), "counts": counts.tolist()},
        }
        _require_json_finite(result, path=f"router_probability_distribution.{name}")
        return result

    result = {
        "token_count": int(normalized.shape[0]),
        "top1_weight": summarize(top_one, "top1_weight"),
        "top1_top2_margin": summarize(margin, "top1_top2_margin"),
        "normalized_entropy": summarize(entropy, "normalized_entropy"),
    }
    _require_json_finite(result, path="router_probability_distribution")
    return result


def _routing_prediction_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    actual_probabilities: torch.Tensor,
    *,
    top_k: int,
    include_binary_calibration: bool,
) -> dict[str, float]:
    """Held-out agreement with the model's observed routing decisions and weights."""

    predicted = top_k_expert_predictions(scores, top_k=top_k)
    intersection = (predicted * targets).sum(-1)
    union = (predicted + targets).clamp_max(1).sum(-1).clamp_min(1)
    predicted_count = predicted.sum(-1).clamp_min(1)
    actual_count = targets.sum(-1).clamp_min(1)
    result = {
        "set_precision_at_k": _checked_metric("set_precision_at_k", float((intersection / predicted_count).mean().item()), lower=0.0, upper=1.0),
        "set_recall_at_k": _checked_metric("set_recall_at_k", float((intersection / actual_count).mean().item()), lower=0.0, upper=1.0),
        "set_jaccard_at_k": _checked_metric("set_jaccard_at_k", float((intersection / union).mean().item()), lower=0.0, upper=1.0),
        "macro_f1": macro_f1_expert_routing(predicted, targets),
        **_router_distribution_metrics(scores, actual_probabilities),
    }
    if include_binary_calibration:
        selection_probabilities = torch.sigmoid(scores)
        result["selection_brier_score"] = _checked_metric(
            "selection_brier_score", float(((selection_probabilities - targets.float()) ** 2).mean().item()), lower=0.0, upper=1.0
        )
        result["selection_ece"] = _expected_calibration_error(selection_probabilities, targets)
    return result


def _probe_metrics(
    probe,
    features: torch.Tensor,
    targets: torch.Tensor,
    actual_probabilities: torch.Tensor,
    *,
    top_k: int,
) -> dict[str, float]:
    with torch.no_grad():
        scores = probe(features)
    return _routing_prediction_metrics(
        scores,
        targets,
        actual_probabilities,
        top_k=top_k,
        include_binary_calibration=True,
    )


def _resolve_routerinterp_protocol(config: dict[str, Any], d_model: int) -> dict[str, Any]:
    """Resolve the paper's OLMoE Top-K SAE protocol without hiding deviations.

    The RouterInterp paper specifies the SAE width/sparsity and the token budgets,
    but not optimiser or batch-size hyperparameters. Those remain explicit config
    values and are recorded as implementation choices.
    """

    protocol = str(config.get("protocol", "custom"))
    sae_settings = dict(config.get("sae", {}))
    if protocol != "routerinterp_olmoe_topk":
        return {
            "name": protocol,
            "sae_fit_tokens": config.get("max_sae_fit_tokens", config.get("max_train_tokens")),
            "probe_fit_tokens": config.get("max_probe_fit_tokens", config.get("max_train_tokens")),
            "active_feature_counts": config.get("active_feature_counts", [int(sae_settings.get("k", 64))]),
            "primary_active_features": config.get("primary_active_features"),
            "sae": sae_settings,
            "strict_token_budgets": False,
        }

    # RouterInterp's OLMoE protocol: 16x expansion, Top-K k=32, 100M
    # SAE-fitting tokens, and 1M tokens for the routing-prediction probes.
    expected_features = 16 * d_model
    required = {
        "k": 32,
        "n_features": expected_features,
    }
    for key, expected in required.items():
        actual = sae_settings.get(key, expected)
        if int(actual) != expected:
            raise ValueError(
                f"RouterInterp OLMoE protocol requires sae.{key}={expected}; got {actual}. "
                "Use protocol='custom' for an intentional deviation."
            )
        sae_settings[key] = expected

    expected_counts = [1, 2, 4, 8, 16, 32]
    actual_counts = [int(value) for value in config.get("active_feature_counts", expected_counts)]
    if actual_counts != expected_counts:
        raise ValueError(
            "RouterInterp OLMoE protocol requires active_feature_counts="
            f"{expected_counts}; got {actual_counts}."
        )
    primary_active_features = int(config.get("primary_active_features", 32))
    if primary_active_features != 32:
        raise ValueError(
            "RouterInterp OLMoE protocol uses primary_active_features=32. "
            "Use protocol='custom' for an intentional deviation."
        )
    sae_fit_tokens = int(config.get("max_sae_fit_tokens", 100_000_000))
    probe_fit_tokens = int(config.get("max_probe_fit_tokens", 1_000_000))
    if sae_fit_tokens != 100_000_000 or probe_fit_tokens != 1_000_000:
        raise ValueError(
            "RouterInterp OLMoE protocol requires max_sae_fit_tokens=100000000 "
            "and max_probe_fit_tokens=1000000. Use protocol='custom' for an intentional deviation."
        )
    return {
        "name": protocol,
        "sae_fit_tokens": sae_fit_tokens,
        "probe_fit_tokens": probe_fit_tokens,
        "active_feature_counts": expected_counts,
        "primary_active_features": primary_active_features,
        "sae": sae_settings,
        "strict_token_budgets": bool(config.get("strict_token_budgets", True)),
    }


def run_routerinterp_analysis_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Train per-layer Top-K SAEs and evaluate routing explanations held out.

    The train and evaluation directories must be disjoint captures. SAE fitting,
    rho ranking, and probe fitting use train tokens only. Reconstruction, routing
    recall, and feature coactivation are reported on held-out tokens.
    """

    project_root = Path(config["project_root"]).resolve()
    train_dir = _project_tmp_path(project_root, str(config["train_artifacts_path"]))
    eval_dir = _project_tmp_path(project_root, str(config["eval_artifacts_path"]))
    if train_dir.resolve() == eval_dir.resolve():
        raise ValueError("RouterInterp train and evaluation artifact directories must differ.")
    pretrained_sae_dir = config.get("pretrained_sae_dir")
    if str(config.get("protocol", "custom")) == "routerinterp_olmoe_topk" and bool(
        config.get("strict_token_budgets", True)
    ) and not pretrained_sae_dir:
        required_tokens = int(config.get("max_sae_fit_tokens", 100_000_000))
        available_tokens = _captured_token_count(train_dir)
        if available_tokens < required_tokens:
            raise ValueError(
                "RouterInterp OLMoE protocol requires 100,000,000 SAE-fitting tokens; "
                f"the captured fitting artifacts contain {available_tokens:,}. "
                "Use the custom pilot profile for a smaller study rather than labelling it paper-scale."
            )
    if pretrained_sae_dir:
        pretrained_root = _project_tmp_path(project_root, str(pretrained_sae_dir))
        fit_manifest = json.loads((pretrained_root / "sae_fit_manifest.json").read_text(encoding="utf-8"))
        required_tokens = int(config.get("max_sae_fit_tokens", 100_000_000))
        if bool(config.get("strict_token_budgets", True)) and int(
            fit_manifest.get("trained_tokens_per_layer", 0)
        ) < required_tokens:
            raise ValueError("The streamed SAE checkpoint does not meet the configured SAE token budget.")
    itda_requested = bool(dict(config.get("itda", {})).get("enabled", False))
    if itda_requested and not pretrained_sae_dir:
        raise ValueError("ITDA analysis requires a streamed dictionary checkpoint directory.")
    if itda_requested and not bool(fit_manifest.get("itda", {}).get("enabled", False)):
        raise ValueError("ITDA was requested for analysis but is absent from the streamed fitting manifest.")
    if itda_requested:
        requested_itda_tokens = int(dict(config.get("itda", {})).get("fit_tokens", 1_000_000))
        fitted_itda_tokens = int(fit_manifest.get("itda", {}).get("sampled_tokens_per_layer", 0))
        if fitted_itda_tokens != requested_itda_tokens:
            raise ValueError(
                f"ITDA analysis requires exactly {requested_itda_tokens:,} fitted tokens; "
                f"the streamed dictionary reports {fitted_itda_tokens:,}."
            )
    output_dir = _project_tmp_path(
        project_root,
        str(config.get("output_path", "tmp/routerinterp/analysis")),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(config.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RouterInterp analysis requested CUDA, but CUDA is unavailable.")
    requested_layers = config["layers"]
    if requested_layers == "captured":
        manifest = json.loads((train_dir / "manifest.json").read_text(encoding="utf-8"))
        requested_layers = manifest["config"]["layers"]
    layers = tuple(int(layer) for layer in requested_layers)
    max_eval_tokens = config.get("max_eval_tokens")
    probe_settings = dict(config.get("probe", {}))
    top_rho_count = int(config.get("top_rho_features", 20))
    seed = int(config.get("seed", 0))
    expert_labels = [str(label) for label in config.get("expert_labels", ())]
    results: dict[str, Any] = {
        "format": "mi_lens.routerinterp.analysis.v1",
        "model_label": str(config.get("model_label", "")),
        "expert_labels": expert_labels,
        "layers": {},
    }

    for layer in layers:
        (
            train_x,
            train_selected,
            train_router_probabilities,
            train_token_ids,
            train_previous_token_ids,
            _train_domains,
            _train_languages,
            _train_datasets,
            train_token_count,
            train_num_experts,
        ) = _load_layer_tokens(
            train_dir,
            layer=layer,
            max_tokens=None,
        )
        (
            eval_x,
            eval_selected,
            eval_router_probabilities,
            eval_token_ids,
            eval_previous_token_ids,
            eval_domains,
            eval_languages,
            eval_datasets,
            eval_token_count,
            eval_num_experts,
        ) = _load_layer_tokens(
            eval_dir, layer=layer, max_tokens=None if max_eval_tokens is None else int(max_eval_tokens)
        )
        eval_prompt_records, eval_prompt_rows, eval_token_positions = _load_token_provenance(
            eval_dir,
            max_tokens=None if max_eval_tokens is None else int(max_eval_tokens),
        )
        if eval_prompt_rows.numel() != eval_x.shape[0]:
            raise ValueError("Held-out provenance rows do not align with router activations.")
        if train_x.shape[1] != eval_x.shape[1]:
            raise ValueError(f"Layer {layer} has inconsistent router-input widths across splits.")
        train_top_k = _constant_top_k(train_selected, "Train")
        eval_top_k = _constant_top_k(eval_selected, "Evaluation")
        if train_top_k != eval_top_k:
            raise ValueError(f"Layer {layer} has different routing top-k values across splits.")
        if train_num_experts != eval_num_experts:
            raise ValueError(f"Layer {layer} has inconsistent router expert widths across splits.")
        n_experts = train_num_experts
        if n_experts < 2:
            raise ValueError(f"Layer {layer} has fewer than two observed experts.")
        if expert_labels and len(expert_labels) != n_experts:
            raise ValueError(
                "Configured expert labels do not match the captured router width: "
                f"received {len(expert_labels)} labels for {n_experts} experts."
            )

        protocol = _resolve_routerinterp_protocol(config, d_model=int(train_x.shape[1]))
        sae_fit_tokens = protocol["sae_fit_tokens"]
        probe_fit_tokens = protocol["probe_fit_tokens"]
        if pretrained_sae_dir:
            # The large SAE corpus was consumed by the streamed trainer. These
            # retained train artifacts are only for the 1M-token probe controls.
            sae_fit_tokens = int(fit_manifest["trained_tokens_per_layer"])
            probe_fit_tokens = min(int(probe_fit_tokens), train_token_count)
        elif sae_fit_tokens is not None:
            sae_fit_tokens = min(int(sae_fit_tokens), train_token_count)
        else:
            sae_fit_tokens = train_token_count
        if probe_fit_tokens is not None:
            probe_fit_tokens = min(int(probe_fit_tokens), sae_fit_tokens)
        else:
            probe_fit_tokens = sae_fit_tokens
        if not pretrained_sae_dir and protocol["strict_token_budgets"] and (
            sae_fit_tokens != int(protocol["sae_fit_tokens"])
            or probe_fit_tokens != int(protocol["probe_fit_tokens"])
        ):
            raise ValueError(
                "Captured fitting artifacts do not meet the RouterInterp paper token budgets: "
                f"need {protocol['sae_fit_tokens']:,} SAE-fitting tokens and "
                f"{protocol['probe_fit_tokens']:,} routing-probe tokens; found {train_token_count:,}."
            )
        retained_train_tokens = probe_fit_tokens if pretrained_sae_dir else sae_fit_tokens
        train_x = train_x[:retained_train_tokens]
        train_selected = train_selected[:retained_train_tokens]
        train_router_probabilities = train_router_probabilities[:retained_train_tokens]
        train_token_ids = train_token_ids[:retained_train_tokens]
        train_previous_token_ids = train_previous_token_ids[:retained_train_tokens]
        train_token_count = retained_train_tokens

        train_x = train_x.to(device)
        eval_x = eval_x.to(device)
        train_targets = selected_expert_targets(train_selected.to(device), n_experts)
        eval_targets = selected_expert_targets(eval_selected.to(device), n_experts)
        train_router_probabilities = train_router_probabilities.to(device)
        eval_router_probabilities = eval_router_probabilities.to(device)
        sae_settings = protocol["sae"]
        sae_training_settings = {
            "steps": int(sae_settings.get("steps", 10_000)),
            "batch_size": int(sae_settings.get("batch_size", 1024)),
            "learning_rate": float(sae_settings.get("learning_rate", 3e-4)),
            "seed": seed + layer,
        }
        sae_config = TopKSAEConfig(
            d_model=int(train_x.shape[1]),
            n_features=int(sae_settings.get("n_features", train_x.shape[1] * 4)),
            k=int(sae_settings.get("k", 64)),
        )
        if pretrained_sae_dir:
            sae = TopKSAE.from_pretrained(pretrained_root / f"layer_{layer:02d}" / "topk_sae.pt").to(device)
            if sae.config != sae_config:
                raise ValueError(f"Pretrained SAE at layer {layer} does not match the requested protocol.")
            loss_history = []
        else:
            sae, loss_history = fit_topk_sae(train_x, sae_config, **sae_training_settings)
        with torch.no_grad():
            train_reconstruction, train_features = sae(train_x)
            eval_reconstruction, eval_features = sae(eval_x)
            train_mse = float(torch.nn.functional.mse_loss(train_reconstruction, train_x).item())
            eval_mse = float(torch.nn.functional.mse_loss(eval_reconstruction, eval_x).item())
        _checked_metric("Top-K SAE train reconstruction MSE", train_mse, lower=0.0)
        _checked_metric("Top-K SAE held-out reconstruction MSE", eval_mse, lower=0.0)
        _require_finite_tensor("Top-K SAE train features", train_features)
        _require_finite_tensor("Top-K SAE held-out features", eval_features)

        itda = None
        itda_train_features = None
        itda_eval_features = None
        itda_train_mse = None
        itda_eval_mse = None
        if itda_requested:
            itda_path = pretrained_root / f"layer_{layer:02d}" / "itda.pt"
            if not itda_path.is_file():
                raise FileNotFoundError(f"Missing ITDA checkpoint for layer {layer}: {itda_path}")
            itda = ITDA.from_pretrained(itda_path).to(device)
            if itda.config.d_model != train_x.shape[1]:
                raise ValueError(f"ITDA dictionary width does not match layer {layer} router inputs.")
            with torch.no_grad():
                itda_train_features = itda.encode(train_x)
                itda_eval_features = itda.encode(eval_x)
                itda_train_mse = float(torch.nn.functional.mse_loss(itda.decode(itda_train_features), train_x).item())
                itda_eval_mse = float(torch.nn.functional.mse_loss(itda.decode(itda_eval_features), eval_x).item())
            _checked_metric("ITDA train reconstruction MSE", itda_train_mse, lower=0.0)
            _checked_metric("ITDA held-out reconstruction MSE", itda_eval_mse, lower=0.0)
            _require_finite_tensor("ITDA train features", itda_train_features)
            _require_finite_tensor("ITDA held-out features", itda_eval_features)
        probe_train_features = train_features[:probe_fit_tokens].detach()
        probe_train_targets = train_targets[:probe_fit_tokens]
        probe_train_token_ids = train_token_ids[:probe_fit_tokens]
        probe_train_previous_token_ids = train_previous_token_ids[:probe_fit_tokens]
        rho = rho_usefulness(probe_train_features, probe_train_targets)
        rho_indices, rho_values = top_rho_features(rho, n_features=top_rho_count)

        pca_components = min(
            int(sae_settings.get("pca_components", train_x.shape[1])),
            int(probe_train_features.shape[0]),
            int(train_x.shape[1]),
        )
        if pca_components < sae_config.k:
            raise ValueError("`pca_components` must be at least the SAE sparsity `k`.")
        # PCA is fitted only on the same fitting rows available to every
        # routing-prediction control, never on held-out activations.
        pca_fit_x = train_x[:probe_fit_tokens]
        pca_mean = pca_fit_x.mean(0, keepdim=True)
        _, _, pca_vectors = torch.pca_lowrank(pca_fit_x - pca_mean, q=pca_components, center=False)
        pca_train = (pca_fit_x - pca_mean) @ pca_vectors
        pca_eval = (eval_x - pca_mean) @ pca_vectors

        unigram_counts, global_counts = _fit_ngram_counts(probe_train_token_ids, probe_train_targets.cpu())
        bigram_train_keys = torch.stack((probe_train_previous_token_ids, probe_train_token_ids), dim=1)
        bigram_eval_keys = torch.stack((eval_previous_token_ids, eval_token_ids), dim=1)
        bigram_counts, _ = _fit_ngram_counts(bigram_train_keys, probe_train_targets.cpu())
        unigram_scores = _predict_ngram_scores(eval_token_ids, unigram_counts, global_counts).to(device)
        bigram_scores = _predict_ngram_scores(bigram_eval_keys, bigram_counts, global_counts).to(device)
        unigram_metrics = _routing_prediction_metrics(
            unigram_scores,
            eval_targets,
            eval_router_probabilities,
            top_k=eval_top_k,
            include_binary_calibration=False,
        )
        bigram_metrics = _routing_prediction_metrics(
            bigram_scores,
            eval_targets,
            eval_router_probabilities,
            top_k=eval_top_k,
            include_binary_calibration=False,
        )

        active_feature_counts = [int(value) for value in protocol["active_feature_counts"]]
        if any(value < 1 or value > sae_config.k for value in active_feature_counts):
            raise ValueError(
                "Every active_feature_counts value must be in [1, sae.k]; "
                f"got {active_feature_counts} with sae.k={sae_config.k}."
            )
        primary_active_features = int(protocol["primary_active_features"] or max(active_feature_counts))
        if primary_active_features not in active_feature_counts:
            raise ValueError("primary_active_features must be included in active_feature_counts.")
        sae_by_active_features: dict[str, dict[str, float]] = {}
        itda_by_active_features: dict[str, dict[str, float]] = {}
        neuron_by_active_features: dict[str, dict[str, float]] = {}
        pca_by_active_features: dict[str, dict[str, float]] = {}
        primary_sae_probe = None
        for active_features in active_feature_counts:
            # RouterInterp evaluates the m largest SAE activations per token.
            sparse_sae_train = sparse_by_magnitude(probe_train_features, k=active_features)
            sparse_sae_eval = sparse_by_magnitude(eval_features, k=active_features)
            probe = fit_routing_probe(
                sparse_sae_train,
                probe_train_targets,
                steps=int(probe_settings.get("steps", 500)),
                learning_rate=float(probe_settings.get("learning_rate", 1e-3)),
            )
            sae_by_active_features[str(active_features)] = _probe_metrics(
                probe, sparse_sae_eval, eval_targets, eval_router_probabilities, top_k=eval_top_k
            )
            if active_features == primary_active_features:
                primary_sae_probe = probe

            if itda is not None and itda_train_features is not None and itda_eval_features is not None:
                if active_features > itda.n_atoms:
                    raise ValueError(
                        f"ITDA layer {layer} has {itda.n_atoms} atoms but comparison requests {active_features}."
                    )
                sparse_itda_train = sparse_by_magnitude(
                    itda_train_features[:probe_fit_tokens], k=active_features
                )
                sparse_itda_eval = sparse_by_magnitude(itda_eval_features, k=active_features)
                itda_probe = fit_routing_probe(
                    sparse_itda_train,
                    probe_train_targets,
                    steps=int(probe_settings.get("steps", 500)),
                    learning_rate=float(probe_settings.get("learning_rate", 1e-3)),
                )
                itda_by_active_features[str(active_features)] = _probe_metrics(
                    itda_probe, sparse_itda_eval, eval_targets, eval_router_probabilities, top_k=eval_top_k
                )

            # These are additional controls, not claimed RouterInterp baselines:
            # both retain exactly the same m coordinates as the SAE representation.
            sparse_neuron_train = sparse_by_magnitude(train_x[:probe_fit_tokens], k=active_features)
            sparse_neuron_eval = sparse_by_magnitude(eval_x, k=active_features)
            neuron_probe = fit_routing_probe(
                sparse_neuron_train,
                probe_train_targets,
                steps=int(probe_settings.get("steps", 500)),
                learning_rate=float(probe_settings.get("learning_rate", 1e-3)),
            )
            neuron_by_active_features[str(active_features)] = _probe_metrics(
                neuron_probe, sparse_neuron_eval, eval_targets, eval_router_probabilities, top_k=eval_top_k
            )

            sparse_pca_train = sparse_by_magnitude(pca_train, k=active_features)
            sparse_pca_eval = sparse_by_magnitude(pca_eval, k=active_features)
            pca_probe = fit_routing_probe(
                sparse_pca_train,
                probe_train_targets,
                steps=int(probe_settings.get("steps", 500)),
                learning_rate=float(probe_settings.get("learning_rate", 1e-3)),
            )
            pca_by_active_features[str(active_features)] = _probe_metrics(
                pca_probe, sparse_pca_eval, eval_targets, eval_router_probabilities, top_k=eval_top_k
            )
        assert primary_sae_probe is not None
        coactivation = feature_coactivation_ratio(eval_features, rho_indices)
        activation_diagnostics = feature_activation_diagnostics(eval_features)
        top_rho_contexts = _top_rho_feature_contexts(
            features=eval_features.cpu(),
            rho_indices=rho_indices.cpu(),
            rho_values=rho_values.cpu(),
            expert_labels=expert_labels,
            prompt_records=eval_prompt_records,
            prompt_rows=eval_prompt_rows,
            token_positions=eval_token_positions,
            max_contexts_per_feature=int(config.get("max_contexts_per_feature", 10)),
        )
        top_rho_domain_profiles = _top_rho_feature_domain_profiles(
            features=eval_features.cpu(),
            domains=eval_domains,
            rho_indices=rho_indices.cpu(),
            rho_values=rho_values.cpu(),
            expert_labels=expert_labels,
        )
        itda_analysis = None
        if itda is not None and itda_train_features is not None and itda_eval_features is not None:
            itda_rho = rho_usefulness(itda_train_features[:probe_fit_tokens].detach(), probe_train_targets)
            itda_rho_indices, itda_rho_values = top_rho_features(itda_rho, n_features=top_rho_count)
            itda_analysis = {
                "config": asdict(itda.config),
                "atom_count": itda.n_atoms,
                "atom_provenance_checkpoint": str(itda_path),
                "train_reconstruction_mse": itda_train_mse,
                "heldout_reconstruction_mse": itda_eval_mse,
                "feature_coactivation": feature_coactivation_ratio(itda_eval_features, itda_rho_indices),
                "feature_activation_diagnostics": feature_activation_diagnostics(itda_eval_features),
                "top_rho_features": [
                    {
                        "expert": expert,
                        "expert_label": expert_labels[expert] if expert_labels else f"expert_{expert}",
                        "feature_ids": itda_rho_indices[expert].cpu().tolist(),
                        "rho": itda_rho_values[expert].cpu().tolist(),
                    }
                    for expert in range(n_experts)
                ],
                "top_rho_contexts": _top_rho_feature_contexts(
                    features=itda_eval_features.cpu(),
                    rho_indices=itda_rho_indices.cpu(),
                    rho_values=itda_rho_values.cpu(),
                    expert_labels=expert_labels,
                    prompt_records=eval_prompt_records,
                    prompt_rows=eval_prompt_rows,
                    token_positions=eval_token_positions,
                    max_contexts_per_feature=int(config.get("max_contexts_per_feature", 10)),
                ),
                "top_rho_domain_profiles": _top_rho_feature_domain_profiles(
                    features=itda_eval_features.cpu(),
                    domains=eval_domains,
                    rho_indices=itda_rho_indices.cpu(),
                    rho_values=itda_rho_values.cpu(),
                    expert_labels=expert_labels,
                ),
            }
        domain_routing = _domain_routing_summary(
            eval_selected,
            eval_domains,
            num_experts=n_experts,
        )
        actual_domain_expert_activation = _expert_activation_by_group(
            eval_selected,
            eval_domains,
            num_experts=n_experts,
        )
        with torch.no_grad():
            primary_sae_eval = sparse_by_magnitude(eval_features, k=primary_active_features)
            sae_predicted_targets = top_k_expert_predictions(
                primary_sae_probe(primary_sae_eval),
                top_k=eval_top_k,
            ).cpu()
        sae_predicted_domain_expert_activation = _expert_activation_targets_by_group(
            sae_predicted_targets,
            eval_domains,
        )
        language_routing = _expert_activation_by_group(
            eval_selected,
            eval_languages,
            num_experts=n_experts,
        )
        language_domain_routing = _expert_activation_by_group(
            eval_selected,
            [f"{language}:{domain}" for language, domain in zip(eval_languages, eval_domains)],
            num_experts=n_experts,
        )
        dataset_routing = _expert_activation_by_group(
            eval_selected,
            eval_datasets,
            num_experts=n_experts,
        )
        geometry_path = pretrained_root / f"layer_{layer:02d}" / "router_geometry.pt" if pretrained_sae_dir else None
        router_geometry = None
        if geometry_path is not None and geometry_path.is_file():
            geometry_payload = torch.load(geometry_path, map_location="cpu", weights_only=True)
            router_geometry = _router_geometry_agreement(
                sae=sae,
                rho=rho.cpu(),
                router_weights=geometry_payload["router_weight"],
                top_n=top_rho_count,
            )

        layer_dir = output_dir / f"layer_{layer:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        sae.save_pretrained(str(layer_dir / "topk_sae.pt"))
        torch.save(
            {
                "probe_state_dict": primary_sae_probe.state_dict(),
                "n_features": sae_config.n_features,
                "n_experts": n_experts,
                "expert_labels": expert_labels,
                "rho": rho.cpu(),
                "top_rho_indices": rho_indices.cpu(),
                "top_rho_values": rho_values.cpu(),
            },
            layer_dir / "routing_analysis.pt",
        )
        _require_json_finite(top_rho_contexts, path=f"layer_{layer}.topk_sae_contexts")
        _require_json_finite(top_rho_domain_profiles, path=f"layer_{layer}.topk_sae_domain_profiles")
        write_json(layer_dir / "topk_sae_top_rho_contexts.json", top_rho_contexts)
        write_json(layer_dir / "topk_sae_top_rho_domain_profiles.json", top_rho_domain_profiles)
        if itda_analysis is not None:
            _require_json_finite(itda_analysis, path=f"layer_{layer}.itda")
            write_json(layer_dir / "itda_analysis.json", itda_analysis)
        layer_result = {
            "sae_config": asdict(sae_config),
            "protocol": protocol,
            "sae_training": sae_training_settings,
            "probe_training": {
                "steps": int(probe_settings.get("steps", 500)),
                "learning_rate": float(probe_settings.get("learning_rate", 1e-3)),
            },
            "train_tokens": train_token_count,
            "sae_fit_tokens": sae_fit_tokens,
            "routing_probe_fit_tokens": probe_fit_tokens,
            "eval_tokens": eval_token_count,
            "num_experts": n_experts,
            "expert_labels": expert_labels,
            "routing_top_k": train_top_k,
            "matched_basis_sparsity": sae_config.k,
            "pca_components": pca_components,
            "pca_fit_tokens": probe_fit_tokens,
            "train_reconstruction_mse": train_mse,
            "heldout_reconstruction_mse": eval_mse,
            "itda": itda_analysis,
            "routing_prediction": {
                "unigram_baseline": unigram_metrics,
                "bigram_baseline": bigram_metrics,
                "sae_predictor": sae_by_active_features[str(primary_active_features)],
                "sae_predictor_by_active_features": sae_by_active_features,
                "itda_predictor": itda_by_active_features.get(str(primary_active_features)),
                "itda_predictor_by_active_features": itda_by_active_features,
                "neuron_basis_probe": neuron_by_active_features[str(primary_active_features)],
                "neuron_basis_probe_by_active_features": neuron_by_active_features,
                "pca_basis_probe": pca_by_active_features[str(primary_active_features)],
                "pca_basis_probe_by_active_features": pca_by_active_features,
            },
            "final_train_loss": loss_history[-1] if loss_history else None,
            "feature_coactivation": coactivation,
            "feature_activation_diagnostics": activation_diagnostics,
            "router_probability_distribution": _router_probability_distribution_summary(
                eval_router_probabilities.cpu()
            ),
            "domain_routing": domain_routing,
            "domain_expert_activation": actual_domain_expert_activation,
            "sae_predicted_domain_expert_activation": sae_predicted_domain_expert_activation,
            "language_routing": language_routing,
            "language_domain_routing": language_domain_routing,
            "dataset_routing": dataset_routing,
            "rho_router_geometry_agreement": router_geometry,
            "top_rho_features": [
                {
                    "expert": expert,
                    "expert_id": expert,
                    "expert_label": expert_labels[expert] if expert_labels else f"expert_{expert}",
                    "feature_ids": rho_indices[expert].cpu().tolist(),
                    "rho": rho_values[expert].cpu().tolist(),
                }
                for expert in range(n_experts)
            ],
            "top_rho_context_artifact": str(layer_dir / "topk_sae_top_rho_contexts.json"),
            "top_rho_domain_profile_artifact": str(layer_dir / "topk_sae_top_rho_domain_profiles.json"),
            "artifact_dir": str(layer_dir),
        }
        _require_json_finite(layer_result, path=f"layer_{layer}")
        write_json(layer_dir / "summary.json", layer_result)
        results["layers"][str(layer)] = layer_result
        del train_x, eval_x, train_features, eval_features, sae, primary_sae_probe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["train_artifacts_path"] = str(train_dir)
    results["eval_artifacts_path"] = str(eval_dir)
    _require_json_finite(results, path="routerinterp_summary")
    write_json(output_dir / "summary.json", results)
    return {"output_dir": str(output_dir), "summary_path": str(output_dir / "summary.json"), **results}
