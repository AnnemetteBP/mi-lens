"""RouterInterp-style SAE analysis of FlexOlmo routing decisions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from mi_lens.adapters.flex_olmo import iter_flex_olmo_layers
from mi_lens.adapters.router_outputs import input_batch_and_sequence_length


@dataclass(slots=True)
class RouterLayerCapture:
    router_input: torch.Tensor
    router_probabilities: torch.Tensor
    selected_experts: torch.Tensor


@dataclass(slots=True)
class RouterInterpCaptureConfig:
    layers: tuple[int, ...]
    max_seq_len: int | None = None
    skip_first_token: bool = True
    artifact_dtype: str = "bfloat16"


def _as_batch_sequence(tensor: torch.Tensor, batch_size: int, sequence_length: int) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor
    if tensor.ndim == 2 and tensor.shape[0] == batch_size * sequence_length:
        return tensor.reshape(batch_size, sequence_length, tensor.shape[-1])
    raise ValueError(
        "Expected router data shaped (batch, token, dim) or (batch * token, dim), "
        f"received {tuple(tensor.shape)}."
    )


def capture_flexolmo_router_layers(
    model,
    inputs: Mapping[str, Any],
    *,
    top_k: int | None = None,
) -> tuple[RouterLayerCapture, ...]:
    """Capture exact pre-gate activations and selected experts for one forward pass."""

    batch_size, sequence_length = input_batch_and_sequence_length(inputs)
    if batch_size is None or sequence_length is None:
        raise ValueError("RouterInterp capture requires batched `input_ids`.")
    raw_inputs: list[torch.Tensor] = []
    raw_outputs: list[Any] = []
    handles = []

    def hook(_module, hook_inputs, output):
        raw_inputs.append(hook_inputs[0].detach())
        raw_outputs.append(output)

    try:
        for layer in iter_flex_olmo_layers(model):
            handles.append(layer.mlp.gate.register_forward_hook(hook))
        with torch.no_grad():
            model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    if not raw_inputs or len(raw_inputs) != len(raw_outputs):
        raise ValueError("No aligned FlexOlmo router inputs and outputs were captured.")

    captures = []
    for router_input, output in zip(raw_inputs, raw_outputs):
        probabilities = output[0] if isinstance(output, tuple) else output
        if not isinstance(probabilities, torch.Tensor):
            raise TypeError("FlexOlmo router did not return a probability tensor.")
        probabilities = _as_batch_sequence(probabilities, batch_size, sequence_length).float()
        router_input = _as_batch_sequence(router_input, batch_size, sequence_length).float()
        if not torch.allclose(
            probabilities.sum(dim=-1),
            torch.ones_like(probabilities[..., 0]),
            rtol=1e-4,
            atol=2e-3,
        ):
            raise ValueError("Expected normalized FlexOlmo router probabilities.")
        effective_top_k = top_k or int(getattr(model.config, "num_experts_per_tok", 1))
        captures.append(
            RouterLayerCapture(
                router_input=router_input,
                router_probabilities=probabilities,
                selected_experts=probabilities.topk(k=effective_top_k, dim=-1).indices,
            )
        )
    return tuple(captures)


def capture_routerinterp_prompt_artifacts(
    model,
    tokenizer,
    examples: Sequence[Mapping[str, Any]],
    *,
    output_dir: str | Path,
    config: RouterInterpCaptureConfig,
) -> dict[str, Any]:
    """Persist the exact RouterInterp training inputs one prompt at a time.

    Each artifact pairs pre-router vectors with that router's probability vector
    and selected expert ids at the same token positions. This avoids reloading a
    55B checkpoint when training or re-evaluating sparse features later.
    """

    dtype_lookup = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        artifact_dtype = dtype_lookup[config.artifact_dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported artifact dtype {config.artifact_dtype!r}.") from exc
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding = model.get_input_embeddings()
    device = embedding.weight.device
    manifest = []
    selected_layers = tuple(sorted(set(int(layer) for layer in config.layers)))

    for prompt_index, example in enumerate(examples):
        prompt = example.get("prompt", example.get("text"))
        if prompt is None:
            raise KeyError("RouterInterp examples require `prompt` or `text`.")
        tokenizer_kwargs = {"return_tensors": "pt", "truncation": config.max_seq_len is not None}
        if config.max_seq_len is not None:
            tokenizer_kwargs["max_length"] = config.max_seq_len
        encoded = tokenizer(str(prompt), **tokenizer_kwargs)
        input_ids = encoded.input_ids.to(device)
        token_count = int(input_ids.shape[1])
        start = 1 if config.skip_first_token else 0
        # RouterInterp explains routing at every real token position. Unlike a
        # next-token lens, it has no reason to discard the final token.
        positions = list(range(start, token_count))
        if not positions:
            continue
        captures = capture_flexolmo_router_layers(model, {"input_ids": input_ids})
        if selected_layers and selected_layers[-1] >= len(captures):
            raise ValueError(f"Requested layer {selected_layers[-1]} but captured {len(captures)} layers.")
        artifact_layers = {}
        for layer in selected_layers:
            capture = captures[layer]
            artifact_layers[str(layer)] = {
                "router_input": capture.router_input[0, positions].cpu().to(artifact_dtype),
                "router_probabilities": capture.router_probabilities[0, positions].cpu().to(artifact_dtype),
                "selected_experts": capture.selected_experts[0, positions].cpu(),
            }
        file_name = f"prompt_{prompt_index:06d}.pt"
        torch.save(
            {
                "prompt_index": prompt_index,
                "example_id": str(example.get("id", example.get("example_id", prompt_index))),
                "prompt": str(prompt),
                "token_ids": input_ids[0].cpu(),
                "positions": torch.tensor(positions, dtype=torch.int32),
                "layers": artifact_layers,
            },
            output_dir / file_name,
        )
        manifest.append(
            {
                "prompt_index": prompt_index,
                "example_id": str(example.get("id", example.get("example_id", prompt_index))),
                "path": file_name,
                "num_tokens": len(positions),
            }
        )

    payload = {
        "format": "mi_lens.routerinterp.v1",
        "config": asdict(config),
        "num_prompts": len(manifest),
        "num_tokens": sum(row["num_tokens"] for row in manifest),
        "prompts": manifest,
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def selected_expert_targets(selected_experts: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Convert top-k ids into multi-label routing targets ``(token, expert)``."""

    if selected_experts.ndim != 2:
        raise ValueError("Expected selected experts with shape (token, top_k).")
    targets = torch.zeros(
        (selected_experts.shape[0], num_experts),
        dtype=torch.float32,
        device=selected_experts.device,
    )
    return targets.scatter_(1, selected_experts, 1.0)


def rho_usefulness(features: torch.Tensor, expert_targets: torch.Tensor) -> torch.Tensor:
    """RouterInterp rho scores with shape ``(expert, feature)``."""

    if features.ndim != 2 or expert_targets.ndim != 2:
        raise ValueError("Features and expert targets must both be two-dimensional.")
    if features.shape[0] != expert_targets.shape[0]:
        raise ValueError("Features and expert targets must have the same token count.")
    scores = []
    for expert_index in range(expert_targets.shape[1]):
        selected = expert_targets[:, expert_index].bool()
        if not selected.any() or selected.all():
            # Small pilot captures may not route any token to every expert.
            # Keep the expert in the fixed router-width output but mark its rho
            # scores as uninformative rather than changing the target space.
            scores.append(torch.zeros(features.shape[1], device=features.device, dtype=features.dtype))
            continue
        scores.append(features[selected].mean(0) - features[~selected].mean(0))
    return torch.stack(scores)


def top_rho_features(rho_scores: torch.Tensor, n_features: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the most routing-predictive SAE latents for each expert."""

    values, indices = rho_scores.topk(k=min(n_features, rho_scores.shape[1]), dim=1)
    return indices, values


class RoutingProbe(nn.Module):
    """Independent logistic classifiers predicting experts from SAE features."""

    def __init__(self, n_features: int, n_experts: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, n_experts)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def fit_routing_probe(
    features: torch.Tensor,
    expert_targets: torch.Tensor,
    *,
    steps: int = 500,
    learning_rate: float = 1e-3,
) -> RoutingProbe:
    """Fit RouterInterp's multi-label logistic routing predictor."""

    probe = RoutingProbe(features.shape[1], expert_targets.shape[1]).to(features.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=1e-4)
    for _ in range(steps):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(probe(features), expert_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return probe.eval()


def routing_recall(probe_logits: torch.Tensor, expert_targets: torch.Tensor, *, top_k: int) -> float:
    """Recall of actual experts among the probe's top-k predictions."""

    predicted = probe_logits.topk(k=top_k, dim=-1).indices
    predicted_targets = selected_expert_targets(predicted, expert_targets.shape[1])
    return float((predicted_targets * expert_targets).sum(-1).div(top_k).mean().item())


def top_k_expert_predictions(scores: torch.Tensor, *, top_k: int) -> torch.Tensor:
    """Convert expert scores into a fixed-cardinality multi-label prediction."""

    if scores.ndim != 2 or not 1 <= top_k <= scores.shape[1]:
        raise ValueError("Scores must be (token, expert) and `top_k` must be valid.")
    return selected_expert_targets(scores.topk(k=top_k, dim=-1).indices, scores.shape[1])


def macro_f1_expert_routing(predicted_targets: torch.Tensor, expert_targets: torch.Tensor) -> float:
    """Macro F1 over binary expert-selection decisions.

    All router experts remain in the macro average, including an expert that is
    unused in a small pilot capture. This makes missing expert coverage visible
    instead of silently changing the denominator.
    """

    if predicted_targets.shape != expert_targets.shape or predicted_targets.ndim != 2:
        raise ValueError("Predicted and true targets must have the same (token, expert) shape.")
    predicted = predicted_targets.bool()
    actual = expert_targets.bool()
    true_positive = (predicted & actual).sum(0).float()
    false_positive = (predicted & ~actual).sum(0).float()
    false_negative = (~predicted & actual).sum(0).float()
    return float(
        (2 * true_positive / (2 * true_positive + false_positive + false_negative).clamp_min(1)).mean().item()
    )


def sparse_by_magnitude(values: torch.Tensor, *, k: int) -> torch.Tensor:
    """Retain the ``k`` strongest signed coordinates of each representation."""

    if values.ndim != 2 or not 1 <= k <= values.shape[1]:
        raise ValueError("Values must be (token, feature) and `k` must be valid.")
    indices = values.abs().topk(k=k, dim=-1).indices
    sparse = torch.zeros_like(values)
    return sparse.scatter(1, indices, values.gather(1, indices))


def feature_coactivation_ratio(features: torch.Tensor, feature_indices_by_expert: torch.Tensor) -> list[dict[str, float | int]]:
    """Paper-style feature co-activation relative to independence for each expert."""

    active = features > 0
    results = []
    for expert_index, indices in enumerate(feature_indices_by_expert):
        selected = active[:, indices]
        pair_ratios = []
        for left in range(selected.shape[1]):
            for right in range(left + 1, selected.shape[1]):
                observed = (selected[:, left] & selected[:, right]).float().mean()
                expected = selected[:, left].float().mean() * selected[:, right].float().mean()
                pair_ratios.append(float((observed / expected.clamp_min(1e-8)).item()))
        results.append(
            {
                "expert": expert_index,
                "median_coactivation_ratio": float(torch.tensor(pair_ratios).median().item()) if pair_ratios else 1.0,
            }
        )
    return results
