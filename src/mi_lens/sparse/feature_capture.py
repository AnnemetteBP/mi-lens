"""Reusable, sharded residual-activation datasets for sparse-feature methods."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


@dataclass(slots=True)
class ResidualCaptureConfig:
    """Controls a durable residual-stream activation capture."""

    layers: tuple[int, ...]
    max_seq_len: int = 512
    start_token_idx: int = 0
    end_token_idx: int | None = None
    skip_first_token: bool = True
    shard_size_tokens: int = 8192
    artifact_dtype: str = "bfloat16"


def _artifact_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float32":
        return torch.float32
    raise ValueError(f"Unsupported artifact dtype {name!r}.")


def _num_layers(model) -> int:
    config = getattr(model, "config", None)
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    num_layers = getattr(text_config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("Model config does not expose `num_hidden_layers`.")
    return int(num_layers)


def _post_block_hidden_states(hidden_states, n_layers: int) -> tuple[torch.Tensor, ...]:
    if hidden_states is None:
        raise ValueError("Model forward pass did not return hidden states.")
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


def _input_device(model) -> torch.device:
    embeddings = model.get_input_embeddings()
    if embeddings is None or not hasattr(embeddings, "weight"):
        raise ValueError("Model does not expose input embedding weights.")
    return embeddings.weight.device


def _example_record(example: str | Mapping[str, Any], example_index: int) -> dict[str, Any]:
    if isinstance(example, str):
        return {"example_id": str(example_index), "prompt": example, "metadata": {}}
    prompt = example.get("prompt", example.get("text"))
    if prompt is None:
        raise KeyError("Each example must contain either `prompt` or `text`.")
    return {
        "example_id": str(example.get("id", example.get("example_id", example_index))),
        "prompt": str(prompt),
        "metadata": {
            key: value
            for key, value in example.items()
            if key not in {"id", "example_id", "prompt", "text"}
        },
    }


def _source_positions(token_count: int, config: ResidualCaptureConfig) -> list[int]:
    # Each stored source token has a known next-token target for later joins.
    last_source_position = token_count - 2
    if last_source_position < 0:
        return []
    start = max(0, int(config.start_token_idx))
    if config.skip_first_token:
        start = max(start, 1)
    end = last_source_position if config.end_token_idx is None else int(config.end_token_idx)
    end = min(end, last_source_position)
    return list(range(start, end + 1)) if end >= start else []


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def capture_residual_activation_shards(
    model,
    tokenizer,
    examples: Sequence[str | Mapping[str, Any]],
    *,
    output_dir: str | Path,
    config: ResidualCaptureConfig,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture selected post-block residuals in compact per-layer shards.

    Each shard stores ``activations`` with shape ``(tokens, d_model)`` plus
    integer joins to the prompt manifest. Full logits are intentionally never
    stored here: they dominate storage and can be recomputed from residuals.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_layers = _num_layers(model)
    layers = tuple(sorted(set(int(layer) for layer in config.layers)))
    if not layers:
        raise ValueError("At least one layer must be selected for residual capture.")
    if layers[0] < 0 or layers[-1] >= n_layers:
        raise ValueError(f"Requested layers {layers} are invalid for {n_layers} layers.")
    if config.shard_size_tokens < 1:
        raise ValueError("`shard_size_tokens` must be positive.")

    dtype = _artifact_dtype(config.artifact_dtype)
    device = _input_device(model)
    layer_dirs = {layer: output_dir / f"layer_{layer:03d}" for layer in layers}
    for layer_dir in layer_dirs.values():
        layer_dir.mkdir(parents=True, exist_ok=True)

    buffers: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    prompt_index_buffer: list[torch.Tensor] = []
    position_buffer: list[torch.Tensor] = []
    token_id_buffer: list[torch.Tensor] = []
    target_token_id_buffer: list[torch.Tensor] = []
    prompt_manifest: list[dict[str, Any]] = []
    shard_manifest: list[dict[str, Any]] = []
    buffered_tokens = 0
    shard_index = 0

    def flush() -> None:
        nonlocal buffered_tokens, shard_index
        if buffered_tokens == 0:
            return
        prompt_indices = torch.cat(prompt_index_buffer)
        positions = torch.cat(position_buffer)
        token_ids = torch.cat(token_id_buffer)
        target_token_ids = torch.cat(target_token_id_buffer)
        for layer in layers:
            activations = torch.cat(buffers[layer]).to(dtype=dtype)
            if activations.shape[0] != buffered_tokens:
                raise AssertionError("Activation and token-index counts must agree.")
            file_name = f"shard_{shard_index:05d}.pt"
            path = layer_dirs[layer] / file_name
            torch.save(
                {
                    "layer": layer,
                    "activations": activations,
                    "prompt_indices": prompt_indices,
                    "token_positions": positions,
                    "token_ids": token_ids,
                    "target_token_ids": target_token_ids,
                },
                path,
            )
            shard_manifest.append(
                {
                    "layer": layer,
                    "path": str(path.relative_to(output_dir)),
                    "shard_index": shard_index,
                    "num_tokens": buffered_tokens,
                    "activation_shape": list(activations.shape),
                    "dtype": str(activations.dtype).replace("torch.", ""),
                }
            )
        for buffer in buffers.values():
            buffer.clear()
        prompt_index_buffer.clear()
        position_buffer.clear()
        token_id_buffer.clear()
        target_token_id_buffer.clear()
        buffered_tokens = 0
        shard_index += 1

    model.eval()
    for example_index, example in enumerate(examples):
        record = _example_record(example, example_index)
        encoded = tokenizer(
            record["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=config.max_seq_len,
        )
        input_ids = encoded.input_ids.to(device)
        positions = _source_positions(int(input_ids.shape[1]), config)
        if not positions:
            continue

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        post_block_states = _post_block_hidden_states(outputs.hidden_states, n_layers)
        source_ids = input_ids[0, positions].detach().cpu()
        target_ids = input_ids[0, [position + 1 for position in positions]].detach().cpu()
        positions_tensor = torch.tensor(positions, dtype=torch.int32)
        prompt_indices = torch.full((len(positions),), example_index, dtype=torch.int32)

        for layer in layers:
            buffers[layer].append(post_block_states[layer][0, positions].detach().cpu())
        prompt_index_buffer.append(prompt_indices)
        position_buffer.append(positions_tensor)
        token_id_buffer.append(source_ids)
        target_token_id_buffer.append(target_ids)
        buffered_tokens += len(positions)
        prompt_manifest.append(
            {
                "prompt_index": example_index,
                "example_id": record["example_id"],
                "prompt": record["prompt"],
                "token_ids": input_ids[0].detach().cpu().tolist(),
                "metadata": record["metadata"],
            }
        )

        if buffered_tokens >= config.shard_size_tokens:
            flush()

    flush()
    _write_jsonl(output_dir / "prompts.jsonl", prompt_manifest)
    manifest = {
        "format": "mi_lens.residual_activation_shards.v1",
        "view": "resid_post",
        "config": asdict(config),
        "n_layers": n_layers,
        "num_prompts": len(prompt_manifest),
        "num_tokens": sum(int(entry["num_tokens"]) for entry in shard_manifest if entry["layer"] == layers[0]),
        "layers": list(layers),
        "shards": shard_manifest,
        "metadata": dict(metadata or {}),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
