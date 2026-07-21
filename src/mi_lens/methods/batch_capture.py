from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from jlens import from_hf

from .fuzzy_trace import FuzzyTraceConfig, evaluate_token_metrics
from .lens_eval import LensEvalConfig, evaluate_surface_metrics


@dataclass(slots=True)
class BatchCaptureConfig:
    max_seq_len: int = 512
    layer_stride: int = 1
    last_n_positions: int = 12
    start_token_idx: int | None = None
    end_token_idx: int | None = None
    top_k: int = 5
    surface_top_k: int = 5
    include_topk_tokens: bool = True
    include_surface_metrics: bool = True
    save_prompt_artifacts: bool = False
    prompt_artifact_dir: str | Path | None = None
    artifact_dtype: str = "float32"


@dataclass(slots=True)
class LensCallResult:
    lens_name: str
    logits_by_layer: dict[int, torch.Tensor]


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _slugify(value: str) -> str:
    out = []
    for ch in str(value):
        out.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    return "".join(out).strip("_") or "item"


def _example_to_record(example: str | Mapping[str, Any], idx: int) -> dict[str, Any]:
    if isinstance(example, str):
        return {
            "example_id": str(idx),
            "prompt_text": example,
            "language": None,
            "meta": {},
        }

    prompt_text = example.get("prompt")
    if prompt_text is None:
        prompt_text = example.get("text")
    if prompt_text is None:
        raise KeyError("Each example must contain either 'prompt' or 'text'.")

    meta = dict(example)
    example_id = _safe_text(meta.pop("id", idx))
    language = meta.get("language") or meta.get("question_language")
    meta.pop("prompt", None)
    meta.pop("text", None)
    return {
        "example_id": example_id,
        "prompt_text": str(prompt_text),
        "language": _safe_text(language),
        "meta": meta,
    }


def _decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode(
        [int(token_id)], clean_up_tokenization_spaces=False
    ).replace("\n", "\\n")


def _positions_from_token_ids(
    token_ids: list[int],
    *,
    start_token_idx: int | None,
    end_token_idx: int | None,
    last_n_positions: int,
) -> list[int]:
    if len(token_ids) < 2:
        raise ValueError("Prompt must tokenize to at least 2 tokens.")

    valid_max = len(token_ids) - 2
    if start_token_idx is not None or end_token_idx is not None:
        start = 0 if start_token_idx is None else start_token_idx
        end = valid_max if end_token_idx is None else end_token_idx
        start = max(0, min(int(start), valid_max))
        end = max(0, min(int(end), valid_max))
        if end < start:
            raise ValueError("end_token_idx must be >= start_token_idx.")
        return list(range(start, end + 1))

    n = max(1, int(last_n_positions))
    return list(range(max(0, (len(token_ids) - 1) - n), len(token_ids) - 1))


def _resolve_layers(
    n_layers: int,
    *,
    jlens_lens=None,
    layer_stride: int = 1,
    layers: Sequence[int] | None = None,
) -> list[int]:
    if layers is not None:
        resolved = sorted(set(int(layer) for layer in layers))
    elif jlens_lens is not None:
        resolved = list(jlens_lens.source_layers)
    else:
        resolved = list(range(n_layers))

    stride = max(1, int(layer_stride))
    resolved = resolved[::stride]
    if not resolved:
        raise ValueError("No layers selected for capture.")
    return resolved


def _normalize_hidden_states(hidden_states, n_layers: int):
    if hidden_states is None:
        raise ValueError("Model forward pass did not return hidden states.")

    total = len(hidden_states)
    if total == n_layers + 1:
        block_outputs = hidden_states[1:]
        tuned_states = hidden_states
    elif total == n_layers + 2:
        block_outputs = hidden_states[1:-1]
        tuned_states = hidden_states[:-1]
    else:
        raise ValueError(
            f"Unexpected hidden-state tuple length {total} for {n_layers} layers."
        )

    if len(block_outputs) != n_layers:
        raise ValueError(
            f"Expected {n_layers} block outputs, found {len(block_outputs)}."
        )
    return block_outputs, tuned_states


def _compute_forward_cache(hf_model, lens_model, prompt_text: str, max_seq_len: int):
    input_ids = lens_model.encode(prompt_text, max_length=max_seq_len)
    with torch.no_grad():
        outputs = hf_model(
            input_ids=input_ids.to(lens_model.input_device),
            output_hidden_states=True,
            use_cache=False,
        )
    token_ids = input_ids[0].tolist()
    block_outputs, tuned_states = _normalize_hidden_states(
        outputs.hidden_states, lens_model.n_layers
    )
    final_logits = outputs.logits[0].detach().float().cpu()
    return {
        "input_ids": input_ids,
        "token_ids": token_ids,
        "block_outputs": block_outputs,
        "tuned_states": tuned_states,
        "final_logits": final_logits,
    }


def _compute_logit_lens_logits(
    lens_model,
    block_outputs,
    layers: Sequence[int],
    positions: Sequence[int],
) -> dict[int, torch.Tensor]:
    logits_by_layer: dict[int, torch.Tensor] = {}
    for layer in layers:
        residual = block_outputs[layer][0][list(positions)].detach().float()
        logits_by_layer[layer] = lens_model.unembed(residual).float().cpu()
    return logits_by_layer


def _compute_tuned_lens_logits(
    tuned_lens,
    tuned_states,
    layers: Sequence[int],
    positions: Sequence[int],
) -> dict[int, torch.Tensor]:
    tuned_device = next(tuned_lens.parameters()).device
    logits_by_layer: dict[int, torch.Tensor] = {}
    for layer in layers:
        tuned_idx = layer + 1
        if tuned_idx >= len(tuned_states):
            raise ValueError(
                f"Tuned-lens hidden-state index {tuned_idx} is out of range."
            )
        hidden = tuned_states[tuned_idx][0][list(positions)].to(tuned_device)
        logits_by_layer[layer] = tuned_lens(hidden, idx=tuned_idx).float().cpu()
    return logits_by_layer


def _compute_tuned_hidden_states(
    tuned_lens,
    tuned_states,
    layers: Sequence[int],
    positions: Sequence[int],
) -> dict[int, torch.Tensor]:
    tuned_device = next(tuned_lens.parameters()).device
    transformed_by_layer: dict[int, torch.Tensor] = {}
    for layer in layers:
        tuned_idx = layer + 1
        if tuned_idx >= len(tuned_states):
            raise ValueError(
                f"Tuned-lens hidden-state index {tuned_idx} is out of range."
            )
        hidden = tuned_states[tuned_idx][0][list(positions)].to(tuned_device)
        transformed_by_layer[layer] = (
            tuned_lens.transform_hidden(hidden, idx=tuned_idx).detach().float().cpu()
        )
    return transformed_by_layer


def _compute_jlens_logits(
    lens_model,
    jlens_lens,
    block_outputs,
    layers: Sequence[int],
    positions: Sequence[int],
) -> dict[int, torch.Tensor]:
    logits_by_layer: dict[int, torch.Tensor] = {}
    for layer in layers:
        if layer not in jlens_lens.source_layers:
            raise ValueError(
                f"Layer L{layer} is not available in the fitted J-lens."
            )
        residual = block_outputs[layer][0][list(positions)].detach().float()
        transported = jlens_lens.transport(residual, layer)
        logits_by_layer[layer] = lens_model.unembed(transported).float().cpu()
    return logits_by_layer


def _compute_jlens_transport(
    jlens_lens,
    block_outputs,
    layers: Sequence[int],
    positions: Sequence[int],
) -> dict[int, torch.Tensor]:
    transported_by_layer: dict[int, torch.Tensor] = {}
    for layer in layers:
        if layer not in jlens_lens.source_layers:
            raise ValueError(
                f"Layer L{layer} is not available in the fitted J-lens."
            )
        residual = block_outputs[layer][0][list(positions)].detach().float()
        transported_by_layer[layer] = jlens_lens.transport(residual, layer).float().cpu()
    return transported_by_layer


def _topk_payload(tokenizer, logits: torch.Tensor, k: int) -> tuple[list[int], list[str]]:
    topk_ids = logits.topk(k).indices.tolist()
    topk_tokens = [_decode_token(tokenizer, token_id) for token_id in topk_ids]
    return [int(token_id) for token_id in topk_ids], topk_tokens


def _artifact_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _coerce_artifact_value(value: Any, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor):
        target_dtype = dtype if torch.is_floating_point(value) else value.dtype
        return value.detach().cpu().to(target_dtype)
    if isinstance(value, dict):
        return {
            key: _coerce_artifact_value(subvalue, dtype)
            for key, subvalue in value.items()
        }
    if isinstance(value, list):
        return [_coerce_artifact_value(item, dtype) for item in value]
    return value


def _write_prompt_artifact(
    artifact_dir: Path,
    artifact_id: str,
    payload: Mapping[str, Any],
    *,
    artifact_dtype_name: str,
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"{artifact_id}.pt"
    torch.save(
        _coerce_artifact_value(dict(payload), _artifact_dtype(artifact_dtype_name)),
        path,
    )
    return path


def capture_lens_batch(
    hf_model,
    tokenizer,
    examples: Sequence[str | Mapping[str, Any]],
    *,
    model_label: str,
    model_variant: str | None = None,
    model_family: str | None = None,
    dataset_name: str | None = None,
    dataset_split_role: str | None = None,
    language: str | None = None,
    lens_model=None,
    jlens_lens=None,
    tuned_lens=None,
    lens_names: Sequence[str] = ("logit", "tuned", "jlens"),
    layers: Sequence[int] | None = None,
    config: BatchCaptureConfig | None = None,
) -> list[dict[str, Any]]:
    """Capture one long-form row per example x position x layer x lens."""

    cfg = config or BatchCaptureConfig()
    if lens_model is None:
        lens_model = from_hf(hf_model, tokenizer)

    requested_lenses = [str(name) for name in lens_names]
    if "tuned" in requested_lenses and tuned_lens is None:
        raise ValueError("Requested tuned-lens capture, but tuned_lens is None.")
    if "jlens" in requested_lenses and jlens_lens is None:
        raise ValueError("Requested J-lens capture, but jlens_lens is None.")
    if cfg.save_prompt_artifacts and cfg.prompt_artifact_dir is None:
        raise ValueError(
            "prompt_artifact_dir must be set when save_prompt_artifacts=True."
        )

    resolved_layers = _resolve_layers(
        lens_model.n_layers,
        jlens_lens=jlens_lens if "jlens" in requested_lenses else None,
        layer_stride=cfg.layer_stride,
        layers=layers,
    )

    token_metric_config = FuzzyTraceConfig(top_k=cfg.top_k)
    surface_metric_config = LensEvalConfig(top_k=cfg.surface_top_k)
    rows: list[dict[str, Any]] = []

    for example_idx, example in enumerate(examples):
        record = _example_to_record(example, example_idx)
        prompt_text = record["prompt_text"]

        cache = _compute_forward_cache(hf_model, lens_model, prompt_text, cfg.max_seq_len)
        positions = _positions_from_token_ids(
            cache["token_ids"],
            start_token_idx=cfg.start_token_idx,
            end_token_idx=cfg.end_token_idx,
            last_n_positions=cfg.last_n_positions,
        )

        final_logits_by_position = cache["final_logits"][positions]
        prompt_source_residuals = {
            int(layer): cache["block_outputs"][layer][0][list(positions)]
            .detach()
            .float()
            .cpu()
            for layer in resolved_layers
        }
        prompt_jlens_transport: dict[int, torch.Tensor] = {}
        prompt_tuned_hidden: dict[int, torch.Tensor] = {}
        lens_results: list[LensCallResult] = []

        if "logit" in requested_lenses:
            lens_results.append(
                LensCallResult(
                    lens_name="logit",
                    logits_by_layer=_compute_logit_lens_logits(
                        lens_model,
                        cache["block_outputs"],
                        resolved_layers,
                        positions,
                    ),
                )
            )
        if "tuned" in requested_lenses:
            prompt_tuned_hidden = _compute_tuned_hidden_states(
                tuned_lens,
                cache["tuned_states"],
                resolved_layers,
                positions,
            )
            lens_results.append(
                LensCallResult(
                    lens_name="tuned",
                    logits_by_layer=_compute_tuned_lens_logits(
                        tuned_lens,
                        cache["tuned_states"],
                        resolved_layers,
                        positions,
                    ),
                )
            )
        if "jlens" in requested_lenses:
            prompt_jlens_transport = _compute_jlens_transport(
                jlens_lens,
                cache["block_outputs"],
                resolved_layers,
                positions,
            )
            lens_results.append(
                LensCallResult(
                    lens_name="jlens",
                    logits_by_layer=_compute_jlens_logits(
                        lens_model,
                        jlens_lens,
                        cache["block_outputs"],
                        resolved_layers,
                        positions,
                    ),
                )
            )

        artifact_id = (
            f"{_slugify(model_label)}__"
            f"{_slugify(dataset_split_role or 'unspecified')}__"
            f"{_slugify(language or record['language'] or 'unknown')}__"
            f"{_slugify(str(record['example_id']))}"
        )
        artifact_path: str | None = None
        if cfg.save_prompt_artifacts:
            prompt_artifact = {
                "artifact_id": artifact_id,
                "model_family": model_family,
                "model_variant": model_variant,
                "model_label": model_label,
                "dataset_name": dataset_name,
                "dataset_split_role": dataset_split_role,
                "language": language or record["language"],
                "example_id": record["example_id"],
                "prompt": prompt_text,
                "token_ids": [int(token_id) for token_id in cache["token_ids"]],
                "decoded_tokens": [
                    _decode_token(tokenizer, int(token_id))
                    for token_id in cache["token_ids"]
                ],
                "positions": [int(position) for position in positions],
                "layers": [int(layer) for layer in resolved_layers],
                "final_logits": final_logits_by_position,
                "final_residual": cache["block_outputs"][lens_model.n_layers - 1][0][
                    list(positions)
                ]
                .detach()
                .float()
                .cpu(),
                "source_residuals": prompt_source_residuals,
                "logits_by_lens": {
                    result.lens_name: {
                        int(layer): tensor for layer, tensor in result.logits_by_layer.items()
                    }
                    for result in lens_results
                },
                "jlens_transported_residuals": prompt_jlens_transport,
                "tuned_transformed_hidden_states": prompt_tuned_hidden,
                "metadata": {
                    "max_seq_len": cfg.max_seq_len,
                    "layer_stride": cfg.layer_stride,
                    "last_n_positions": cfg.last_n_positions,
                    "top_k": cfg.top_k,
                    "artifact_dtype": cfg.artifact_dtype,
                },
            }
            artifact_file = _write_prompt_artifact(
                Path(cfg.prompt_artifact_dir),
                artifact_id,
                prompt_artifact,
                artifact_dtype_name=cfg.artifact_dtype,
            )
            artifact_path = str(artifact_file)

        for position_idx, position in enumerate(positions):
            target_token_id = int(cache["token_ids"][position + 1])
            source_token_id = int(cache["token_ids"][position])
            final_logits = final_logits_by_position[position_idx]
            final_probs = torch.softmax(final_logits.float(), dim=0)
            final_topk_ids = final_logits.topk(cfg.top_k).indices.tolist()
            final_top1_id = int(final_logits.argmax().item())
            final_top1_text = _decode_token(tokenizer, final_top1_id)
            final_gold_rank = int(
                (final_logits.argsort(descending=True) == target_token_id).nonzero()[0].item() + 1
            )

            for lens_result in lens_results:
                for layer, layer_logits in lens_result.logits_by_layer.items():
                    lens_logits = layer_logits[position_idx]
                    lens_probs = torch.softmax(lens_logits.float(), dim=0)
                    token_metrics = evaluate_token_metrics(
                        lens_logits,
                        final_logits,
                        target_token_id,
                        config=token_metric_config,
                    )

                    row = {
                        "model_family": model_family,
                        "model_variant": model_variant,
                        "model_label": model_label,
                        "lens_name": lens_result.lens_name,
                        "dataset_name": dataset_name,
                        "dataset_split_role": dataset_split_role,
                        "language": language or record["language"],
                        "example_id": record["example_id"],
                        "prompt": prompt_text,
                        "layer": int(layer),
                        "position": int(position),
                        "source_token_id": source_token_id,
                        "source_token_text": _decode_token(tokenizer, source_token_id),
                        "target_token_id": target_token_id,
                        "target_token_text": _decode_token(tokenizer, target_token_id),
                        "final_top1_id": final_top1_id,
                        "final_top1_text": final_top1_text,
                        "final_top1_exact": bool(final_top1_id == target_token_id),
                        "final_topk_exact": bool(target_token_id in final_topk_ids),
                        "final_gold_rank": final_gold_rank,
                        "final_gold_prob": float(final_probs[target_token_id].item()),
                        "lens_top1_id": int(lens_logits.argmax().item()),
                        "lens_top1_text": _decode_token(
                            tokenizer, int(lens_logits.argmax().item())
                        ),
                        **asdict(token_metrics),
                        "artifact_id": artifact_id,
                        "artifact_path": artifact_path,
                        "artifact_example_index": int(example_idx),
                        "artifact_position_index": int(position_idx),
                        "artifact_layer_index": int(layer),
                        "artifact_lens_name": lens_result.lens_name,
                        "metric_epsilon": float(token_metric_config.eps),
                        "final_prob_sum": float(final_probs.sum().item()),
                        "lens_prob_sum": float(lens_probs.sum().item()),
                        "final_logits_has_nan": bool(torch.isnan(final_logits).any().item()),
                        "lens_logits_has_nan": bool(torch.isnan(lens_logits).any().item()),
                        "final_probs_has_nan": bool(torch.isnan(final_probs).any().item()),
                        "lens_probs_has_nan": bool(torch.isnan(lens_probs).any().item()),
                    }

                    if cfg.include_surface_metrics:
                        surface_metrics = evaluate_surface_metrics(
                            lens_logits,
                            final_logits,
                            target_token_id,
                            tokenizer=tokenizer,
                            config=surface_metric_config,
                        )
                        row.update(asdict(surface_metrics))

                    if cfg.include_topk_tokens:
                        topk_ids, topk_tokens = _topk_payload(
                            tokenizer, lens_logits, cfg.top_k
                        )
                        row["lens_topk_ids"] = topk_ids
                        row["lens_topk_tokens"] = topk_tokens

                    for key, value in record["meta"].items():
                        if key not in row:
                            row[key] = value

                    rows.append(row)

    return rows


def write_capture_rows_jsonl(rows: Sequence[Mapping[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def summarize_capture_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_by: Sequence[str],
    metric_fields: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    if metric_fields is None:
        sample = rows[0]
        metric_fields = [
            key
            for key, value in sample.items()
            if isinstance(value, (int, float, bool))
            and key
            not in {
                "layer",
                "position",
                "source_token_id",
                "target_token_id",
                "final_top1_id",
                "lens_top1_id",
                "artifact_example_index",
                "artifact_position_index",
                "artifact_layer_index",
            }
        ]

    buckets: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in group_by)
        bucket = buckets.setdefault(
            key,
            {
                "group": {field: row.get(field) for field in group_by},
                "count": 0,
                "sums": defaultdict(float),
            },
        )
        bucket["count"] += 1
        for metric in metric_fields:
            value = row.get(metric)
            if isinstance(value, bool):
                bucket["sums"][metric] += float(value)
            elif isinstance(value, (int, float)):
                bucket["sums"][metric] += float(value)

    summary_rows: list[dict[str, Any]] = []
    for bucket in buckets.values():
        count = bucket["count"]
        summary = dict(bucket["group"])
        summary["count"] = count
        for metric in metric_fields:
            if metric in bucket["sums"]:
                summary[f"{metric}_mean"] = bucket["sums"][metric] / count
        summary_rows.append(summary)

    return summary_rows


def compare_group_means(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    group_by: Sequence[str],
    compare_key: str,
    left_value: str,
    right_value: str,
    metric_fields: Sequence[str],
) -> list[dict[str, Any]]:
    keyed: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in summary_rows:
        key = tuple(row.get(field) for field in (*group_by, compare_key))
        keyed[key] = row

    deltas: list[dict[str, Any]] = []
    groups = {tuple(row.get(field) for field in group_by) for row in summary_rows}
    for group in groups:
        left = keyed.get((*group, left_value))
        right = keyed.get((*group, right_value))
        if left is None or right is None:
            continue

        delta_row = {field: group[idx] for idx, field in enumerate(group_by)}
        delta_row[compare_key] = f"{left_value}__vs__{right_value}"
        for metric in metric_fields:
            left_metric = left.get(f"{metric}_mean")
            right_metric = right.get(f"{metric}_mean")
            if isinstance(left_metric, (int, float)) and isinstance(
                right_metric, (int, float)
            ):
                delta_row[f"{metric}_mean_delta"] = float(left_metric) - float(
                    right_metric
                )
        deltas.append(delta_row)

    return deltas
