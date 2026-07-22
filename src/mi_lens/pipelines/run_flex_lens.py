"""Config-driven FlexLens capture for router-aware logit analysis."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import jlens

from mi_lens.methods.flex_lens import FlexLensConfig, router_metrics, token_metrics
from mi_lens.sparse.router_interp import capture_flexolmo_router_layers

from .common import (
    ModelLoadConfig,
    clear_runtime_state,
    load_jsonl_records,
    load_model_and_tokenizer,
    write_json,
)


def _post_block_states(hidden_states: Sequence[torch.Tensor], n_layers: int) -> tuple[torch.Tensor, ...]:
    if len(hidden_states) == n_layers + 1:
        states = hidden_states[1:]
    elif len(hidden_states) == n_layers + 2:
        states = hidden_states[1:-1]
    else:
        raise ValueError(
            f"Unexpected hidden-state tuple length {len(hidden_states)} for {n_layers} layers."
        )
    if len(states) != n_layers:
        raise ValueError(f"Expected {n_layers} post-block states, found {len(states)}.")
    return tuple(states)


def _decode(tokenizer, token_id: int) -> str:
    return tokenizer.decode(
        [int(token_id)], clean_up_tokenization_spaces=False
    ).replace("\n", "\\n")


def _metric_payload(tokenizer, logits: torch.Tensor, target_id: int, cfg: FlexLensConfig) -> dict[str, Any]:
    metrics = asdict(token_metrics(logits, target_id, config=cfg))
    metrics["top_k_tokens"] = [
        _decode(tokenizer, token_id) for token_id in metrics["top_k_token_ids"]
    ]
    return metrics


def _jsonl_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _capture_prompt(
    model,
    tokenizer,
    lens_model,
    example: Mapping[str, Any],
    *,
    layers: tuple[int, ...],
    max_seq_len: int,
    skip_first_token: bool,
    flex_cfg: FlexLensConfig,
    expert_labels: Sequence[str],
    save_artifact_dir: Path | None,
    prompt_index: int,
) -> list[dict[str, Any]]:
    prompt = example.get("prompt", example.get("text"))
    if prompt is None:
        raise KeyError("FlexLens examples require `prompt` or `text`.")
    encoded = tokenizer(
        str(prompt),
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
    )
    input_ids = encoded.input_ids.to(model.get_input_embeddings().weight.device)
    token_ids = input_ids[0].detach().cpu().tolist()
    positions = list(range(1 if skip_first_token else 0, len(token_ids) - 1))
    if not positions:
        return []

    router_captures = capture_flexolmo_router_layers(
        model,
        {"input_ids": input_ids},
        capture_mixture_output=True,
    )
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    block_states = _post_block_states(outputs.hidden_states, len(router_captures))
    final_logits = outputs.logits[0, positions].detach().float().cpu()
    target_ids = torch.tensor(
        [token_ids[position + 1] for position in positions], dtype=torch.long
    )
    prompt_id = str(example.get("id", example.get("example_id", prompt_index)))
    dataset_name = str(example.get("dataset_name", example.get("task", "unknown")))
    language = str(example.get("language", example.get("question_language", "unknown")))
    domain = str(example.get("domain", example.get("task", "unknown")))

    if save_artifact_dir is not None:
        save_artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "mi_lens.flexlens.prompt.v1",
                "prompt_index": prompt_index,
                "example_id": prompt_id,
                "dataset_name": dataset_name,
                "language": language,
                "domain": domain,
                "prompt": str(prompt),
                "token_ids": torch.tensor(token_ids, dtype=torch.long),
                "target_token_ids": target_ids,
                "positions": torch.tensor(positions, dtype=torch.int32),
                "layers": {
                    str(layer): {
                        "router_input": router_captures[layer].router_input[0, positions].cpu().to(torch.bfloat16),
                        "router_scores": router_captures[layer].router_scores[0, positions].cpu().to(torch.bfloat16),
                        "router_probabilities": router_captures[layer].router_probabilities[0, positions].cpu().to(torch.bfloat16),
                        "selected_experts": router_captures[layer].selected_experts[0, positions].cpu(),
                        "selected_weights": router_captures[layer].selected_weights[0, positions].cpu().to(torch.bfloat16),
                        "post_block_state": block_states[layer][0, positions].detach().cpu().to(torch.bfloat16),
                        "mixture_output": router_captures[layer].mixture_output[0, positions].cpu().to(torch.bfloat16),
                    }
                    for layer in layers
                },
            },
            save_artifact_dir / f"prompt_{prompt_index:06d}.pt",
        )

    rows: list[dict[str, Any]] = []
    for layer in layers:
        capture = router_captures[layer]
        probabilities = capture.router_probabilities[0, positions].float().cpu()
        if expert_labels and probabilities.shape[-1] != len(expert_labels):
            raise ValueError(
                f"Layer {layer} has {probabilities.shape[-1]} experts but "
                f"{len(expert_labels)} labels were configured."
            )
        router_summary = router_metrics(probabilities, config=flex_cfg)
        pre_state = capture.router_input[0, positions].float()
        post_state = block_states[layer][0, positions].detach().float()
        mixture_state = capture.mixture_output[0, positions].float()
        pre_logits = lens_model.unembed(pre_state).float().cpu()
        post_logits = lens_model.unembed(post_state).float().cpu()
        mixture_logits = lens_model.unembed(mixture_state).float().cpu()
        for offset, position in enumerate(positions):
            top_experts = capture.selected_experts[0, position].detach().cpu().tolist()
            expert_names = [expert_labels[int(expert)] for expert in top_experts] if expert_labels else []
            row = {
                "format": "mi_lens.flexlens.row.v1",
                "prompt_index": prompt_index,
                "example_id": prompt_id,
                "dataset_name": dataset_name,
                "language": language,
                "domain": domain,
                "position": int(position),
                "source_token_id": int(token_ids[position]),
                "source_token": _decode(tokenizer, token_ids[position]),
                "target_token_id": int(target_ids[offset]),
                "target_token": _decode(tokenizer, token_ids[position + 1]),
                "layer": int(layer),
                "router_top_expert": int(router_summary["top1_expert"][offset].item()),
                "router_top_expert_name": (
                    expert_labels[int(router_summary["top1_expert"][offset].item())]
                    if expert_labels else None
                ),
                "router_selected_experts": [int(expert) for expert in top_experts],
                "router_selected_expert_names": expert_names,
                "router_selected_weights": [
                    float(value) for value in capture.selected_weights[0, position].detach().cpu().tolist()
                ],
                "router_entropy": float(router_summary["entropy"][offset].item()),
                "router_top1_probability": float(router_summary["top1_probability"][offset].item()),
                "router_top2_probability": float(router_summary["top2_probability"][offset].item()),
                "router_top1_top2_margin": float(router_summary["top1_top2_margin"][offset].item()),
                "pre_router": _metric_payload(tokenizer, pre_logits[offset], int(target_ids[offset]), flex_cfg),
                "post_block": _metric_payload(tokenizer, post_logits[offset], int(target_ids[offset]), flex_cfg),
                "expert_mixture_readout": _metric_payload(tokenizer, mixture_logits[offset], int(target_ids[offset]), flex_cfg),
                "final": _metric_payload(tokenizer, final_logits[offset], int(target_ids[offset]), flex_cfg),
                "post_minus_pre_target_logit": float(
                    post_logits[offset, int(target_ids[offset])] - pre_logits[offset, int(target_ids[offset])]
                ),
                "final_minus_post_target_logit": float(
                    final_logits[offset, int(target_ids[offset])] - post_logits[offset, int(target_ids[offset])]
                ),
            }
            rows.append(row)
    return rows


def run_flex_lens_pipeline(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run FlexLens on one composed model and persist compact token rows."""

    project_root = Path(config["project_root"])
    model_cfg = ModelLoadConfig(**dict(config["model"]))
    configured_example_paths = config.get("examples_paths")
    if configured_example_paths:
        examples = []
        for examples_path in configured_example_paths:
            examples.extend(load_jsonl_records(project_root / str(examples_path)))
    else:
        examples = load_jsonl_records(project_root / str(config["examples_path"]))
    max_examples = config.get("max_examples")
    if max_examples is not None:
        examples = examples[: int(max_examples)]
    layers = tuple(sorted(set(int(layer) for layer in config["layers"])))
    if not layers:
        raise ValueError("FlexLens requires at least one layer.")
    flex_cfg = FlexLensConfig(
        top_k=int(config.get("top_k", 8)),
        eps=float(config.get("eps", 1e-12)),
    )
    output_dir = project_root / str(config.get("output_dir", "outputs/flexlens"))
    model_label = str(config.get("model_label", model_cfg.model_name))
    model_dir = output_dir / model_label
    row_path = model_dir / "rows.jsonl"
    artifact_dir = model_dir / "prompts" if config.get("save_prompt_artifacts", True) else None

    hf_model, tokenizer, cache_dir = load_model_and_tokenizer(
        model_cfg,
        project_root=project_root,
    )
    hf_model.eval()
    lens_model = jlens.from_hf(hf_model, tokenizer)
    expert_labels = tuple(str(label) for label in config.get("expert_labels", []))
    rows: list[dict[str, Any]] = []
    for prompt_index, example in enumerate(examples):
        rows.extend(
            _capture_prompt(
                hf_model,
                tokenizer,
                lens_model,
                example,
                layers=layers,
                max_seq_len=int(config.get("max_seq_len", 2048)),
                skip_first_token=bool(config.get("skip_first_token", False)),
                flex_cfg=flex_cfg,
                expert_labels=expert_labels,
                save_artifact_dir=artifact_dir,
                prompt_index=prompt_index,
            )
        )
    _jsonl_write(row_path, rows)
    metadata = {
        "format": "mi_lens.flexlens.run.v1",
        "model_label": model_label,
        "model_config": dict(config["model"]),
        "layers": list(layers),
        "num_prompts": len(examples),
        "num_rows": len(rows),
        "row_path": str(row_path),
        "prompt_artifact_dir": str(artifact_dir) if artifact_dir else None,
        "cache_dir": str(cache_dir),
        "notes": {
            "full_logits_saved": False,
            "expert_mixture_readout_is_not_a_residual_state": True,
            "expert_labels": list(expert_labels),
        },
    }
    write_json(model_dir / "metadata.json", metadata)
    clear_runtime_state(None, None, lens_model, hf_model, tokenizer)
    return metadata
