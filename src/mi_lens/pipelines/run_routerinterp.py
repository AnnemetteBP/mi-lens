"""End-to-end capture and held-out analysis for RouterInterp-style SAEs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from mi_lens.sparse import (
    RouterInterpCaptureConfig,
    TopKSAEConfig,
    capture_routerinterp_prompt_artifacts,
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

from .common import (
    ModelLoadConfig,
    clear_runtime_state,
    load_jsonl_records,
    load_model_and_tokenizer,
    metadata_payload,
    slugify,
    write_json,
)


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


def run_routerinterp_capture_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Capture pre-router vectors and routing labels from a fixed JSONL split."""

    project_root = Path(config["project_root"]).resolve()
    model_config = ModelLoadConfig(**config["model"])
    output_dir = _routerinterp_output_dir(config)
    records = load_jsonl_records(project_root / config["examples_path"])
    if config.get("max_examples") is not None:
        records = records[: int(config["max_examples"])]
    capture_config = RouterInterpCaptureConfig(
        layers=tuple(int(layer) for layer in config["layers"]),
        max_seq_len=int(config.get("max_seq_len", 512)),
        skip_first_token=bool(config.get("skip_first_token", True)),
        artifact_dtype=str(config.get("artifact_dtype", "bfloat16")),
    )

    model, tokenizer, cache_dir = load_model_and_tokenizer(model_config, project_root=project_root)
    model.eval()
    try:
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
                    "examples_path": str(config["examples_path"]),
                    "dataset_split_role": config.get("dataset_split_role", "train"),
                    "layers": list(capture_config.layers),
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Read an explicit token cap from per-prompt artifacts in manifest order."""

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    activations: list[torch.Tensor] = []
    selected: list[torch.Tensor] = []
    token_ids: list[torch.Tensor] = []
    previous_token_ids: list[torch.Tensor] = []
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
        probabilities = block["router_probabilities"]
        positions = payload["positions"].long()
        full_token_ids = payload["token_ids"].long()
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError(f"Malformed RouterInterp artifact {artifact_dir / row['path']}.")
        if probabilities.ndim != 2 or probabilities.shape[0] != x.shape[0]:
            raise ValueError(f"Malformed router probabilities in {artifact_dir / row['path']}.")
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
            x, y = x[:remaining], y[:remaining]
            source_ids, previous_ids = source_ids[:remaining], previous_ids[:remaining]
        activations.append(x)
        selected.append(y)
        token_ids.append(source_ids)
        previous_token_ids.append(previous_ids)
        retained += x.shape[0]
    if not activations:
        raise ValueError(f"No tokens available for layer {layer} in {artifact_dir}.")
    assert num_experts is not None
    return (
        torch.cat(activations),
        torch.cat(selected),
        torch.cat(token_ids),
        torch.cat(previous_token_ids),
        retained,
        num_experts,
    )


def _constant_top_k(selected_experts: torch.Tensor, split_name: str) -> int:
    if selected_experts.ndim != 2 or selected_experts.shape[1] < 1:
        raise ValueError(f"{split_name} selected-expert targets must be shaped (token, top_k).")
    return int(selected_experts.shape[1])


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


def _probe_metrics(probe, features: torch.Tensor, targets: torch.Tensor, *, top_k: int) -> dict[str, float]:
    with torch.no_grad():
        scores = probe(features)
        predicted = top_k_expert_predictions(scores, top_k=top_k)
    return {
        "set_recall_at_k": routing_recall(scores, targets, top_k=top_k),
        "macro_f1": macro_f1_expert_routing(predicted, targets),
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
    output_dir = _project_tmp_path(
        project_root,
        str(config.get("output_path", "tmp/routerinterp/analysis")),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(config.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RouterInterp analysis requested CUDA, but CUDA is unavailable.")
    layers = tuple(int(layer) for layer in config["layers"])
    max_train_tokens = config.get("max_train_tokens")
    max_eval_tokens = config.get("max_eval_tokens")
    sae_settings = dict(config.get("sae", {}))
    probe_settings = dict(config.get("probe", {}))
    top_rho_count = int(config.get("top_rho_features", 20))
    seed = int(config.get("seed", 0))
    results: dict[str, Any] = {"format": "mi_lens.routerinterp.analysis.v1", "layers": {}}

    for layer in layers:
        (
            train_x,
            train_selected,
            train_token_ids,
            train_previous_token_ids,
            train_token_count,
            train_num_experts,
        ) = _load_layer_tokens(
            train_dir, layer=layer, max_tokens=None if max_train_tokens is None else int(max_train_tokens)
        )
        (
            eval_x,
            eval_selected,
            eval_token_ids,
            eval_previous_token_ids,
            eval_token_count,
            eval_num_experts,
        ) = _load_layer_tokens(
            eval_dir, layer=layer, max_tokens=None if max_eval_tokens is None else int(max_eval_tokens)
        )
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

        train_x = train_x.to(device)
        eval_x = eval_x.to(device)
        train_targets = selected_expert_targets(train_selected.to(device), n_experts)
        eval_targets = selected_expert_targets(eval_selected.to(device), n_experts)
        sae_config = TopKSAEConfig(
            d_model=int(train_x.shape[1]),
            n_features=int(sae_settings.get("n_features", train_x.shape[1] * 4)),
            k=int(sae_settings.get("k", 64)),
        )
        sae, loss_history = fit_topk_sae(
            train_x,
            sae_config,
            steps=int(sae_settings.get("steps", 10_000)),
            batch_size=int(sae_settings.get("batch_size", 1024)),
            learning_rate=float(sae_settings.get("learning_rate", 3e-4)),
            seed=seed + layer,
        )
        with torch.no_grad():
            train_reconstruction, train_features = sae(train_x)
            eval_reconstruction, eval_features = sae(eval_x)
            train_mse = float(torch.nn.functional.mse_loss(train_reconstruction, train_x).item())
            eval_mse = float(torch.nn.functional.mse_loss(eval_reconstruction, eval_x).item())
        rho = rho_usefulness(train_features, train_targets)
        rho_indices, rho_values = top_rho_features(rho, n_features=top_rho_count)
        probe = fit_routing_probe(
            train_features.detach(),
            train_targets,
            steps=int(probe_settings.get("steps", 500)),
            learning_rate=float(probe_settings.get("learning_rate", 1e-3)),
        )
        sae_probe_metrics = _probe_metrics(probe, eval_features, eval_targets, top_k=eval_top_k)

        # Matched sparse-basis ablations: raw residual coordinates and PCA
        # coordinates each retain the same number of active values as the SAE.
        sparse_neuron_train = sparse_by_magnitude(train_x, k=sae_config.k)
        sparse_neuron_eval = sparse_by_magnitude(eval_x, k=sae_config.k)
        neuron_probe = fit_routing_probe(
            sparse_neuron_train,
            train_targets,
            steps=int(probe_settings.get("steps", 500)),
            learning_rate=float(probe_settings.get("learning_rate", 1e-3)),
        )
        neuron_probe_metrics = _probe_metrics(neuron_probe, sparse_neuron_eval, eval_targets, top_k=eval_top_k)

        pca_components = min(
            int(sae_settings.get("pca_components", min(train_x.shape[1], 1024))),
            int(train_x.shape[0]),
            int(train_x.shape[1]),
        )
        if pca_components < sae_config.k:
            raise ValueError("`pca_components` must be at least the SAE sparsity `k`.")
        pca_mean = train_x.mean(0, keepdim=True)
        _, _, pca_vectors = torch.pca_lowrank(train_x - pca_mean, q=pca_components, center=False)
        sparse_pca_train = sparse_by_magnitude((train_x - pca_mean) @ pca_vectors, k=sae_config.k)
        sparse_pca_eval = sparse_by_magnitude((eval_x - pca_mean) @ pca_vectors, k=sae_config.k)
        pca_probe = fit_routing_probe(
            sparse_pca_train,
            train_targets,
            steps=int(probe_settings.get("steps", 500)),
            learning_rate=float(probe_settings.get("learning_rate", 1e-3)),
        )
        pca_probe_metrics = _probe_metrics(pca_probe, sparse_pca_eval, eval_targets, top_k=eval_top_k)

        unigram_counts, global_counts = _fit_ngram_counts(train_token_ids, train_targets.cpu())
        bigram_train_keys = torch.stack((train_previous_token_ids, train_token_ids), dim=1)
        bigram_eval_keys = torch.stack((eval_previous_token_ids, eval_token_ids), dim=1)
        bigram_counts, _ = _fit_ngram_counts(bigram_train_keys, train_targets.cpu())
        unigram_scores = _predict_ngram_scores(eval_token_ids, unigram_counts, global_counts).to(device)
        bigram_scores = _predict_ngram_scores(bigram_eval_keys, bigram_counts, global_counts).to(device)
        unigram_predictions = top_k_expert_predictions(unigram_scores, top_k=eval_top_k)
        bigram_predictions = top_k_expert_predictions(bigram_scores, top_k=eval_top_k)
        unigram_metrics = {
            "set_recall_at_k": routing_recall(unigram_scores, eval_targets, top_k=eval_top_k),
            "macro_f1": macro_f1_expert_routing(unigram_predictions, eval_targets),
        }
        bigram_metrics = {
            "set_recall_at_k": routing_recall(bigram_scores, eval_targets, top_k=eval_top_k),
            "macro_f1": macro_f1_expert_routing(bigram_predictions, eval_targets),
        }
        coactivation = feature_coactivation_ratio(eval_features, rho_indices)

        layer_dir = output_dir / f"layer_{layer:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        sae.save_pretrained(str(layer_dir / "topk_sae.pt"))
        torch.save(
            {
                "probe_state_dict": probe.state_dict(),
                "n_features": sae_config.n_features,
                "n_experts": n_experts,
                "rho": rho.cpu(),
                "top_rho_indices": rho_indices.cpu(),
                "top_rho_values": rho_values.cpu(),
            },
            layer_dir / "routing_analysis.pt",
        )
        layer_result = {
            "sae_config": asdict(sae_config),
            "train_tokens": train_token_count,
            "eval_tokens": eval_token_count,
            "num_experts": n_experts,
            "routing_top_k": train_top_k,
            "matched_basis_sparsity": sae_config.k,
            "pca_components": pca_components,
            "train_reconstruction_mse": train_mse,
            "heldout_reconstruction_mse": eval_mse,
            "routing_prediction": {
                "unigram_baseline": unigram_metrics,
                "bigram_baseline": bigram_metrics,
                "neuron_basis_probe": neuron_probe_metrics,
                "pca_basis_probe": pca_probe_metrics,
                "sae_predictor": sae_probe_metrics,
            },
            "final_train_loss": loss_history[-1],
            "feature_coactivation": coactivation,
            "top_rho_features": [
                {
                    "expert": expert,
                    "feature_ids": rho_indices[expert].cpu().tolist(),
                    "rho": rho_values[expert].cpu().tolist(),
                }
                for expert in range(n_experts)
            ],
            "artifact_dir": str(layer_dir),
        }
        write_json(layer_dir / "summary.json", layer_result)
        results["layers"][str(layer)] = layer_result
        del train_x, eval_x, train_features, eval_features, sae, probe, neuron_probe, pca_probe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["train_artifacts_path"] = str(train_dir)
    results["eval_artifacts_path"] = str(eval_dir)
    write_json(output_dir / "summary.json", results)
    return {"output_dir": str(output_dir), "summary_path": str(output_dir / "summary.json"), **results}
