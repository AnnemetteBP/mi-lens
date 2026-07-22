"""RouterInterp-style SAE analysis of FlexOlmo routing decisions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from mi_lens.adapters.flex_olmo import iter_flex_olmo_layers
from mi_lens.adapters.router_outputs import (
    input_batch_and_sequence_length,
    router_scores_to_probabilities,
)


def _require_finite(name: str, values: torch.Tensor) -> None:
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinity.")


def _require_probability_rows(name: str, probabilities: torch.Tensor) -> None:
    """Reject malformed router distributions instead of silently normalising them."""

    if probabilities.ndim < 2 or probabilities.shape[-1] < 1:
        raise ValueError(f"{name} must have a non-empty expert dimension.")
    _require_finite(name, probabilities)
    tolerance = 2e-3 if probabilities.dtype in (torch.float16, torch.bfloat16) else 1e-5
    if (probabilities < -tolerance).any() or (probabilities > 1.0 + tolerance).any():
        raise ValueError(f"{name} contains values outside [0, 1].")
    sums = probabilities.float().sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), rtol=tolerance, atol=tolerance):
        raise ValueError(f"{name} rows are not normalised probability distributions.")


@dataclass(slots=True)
class RouterLayerCapture:
    router_input: torch.Tensor
    router_scores: torch.Tensor
    router_probabilities: torch.Tensor
    selected_experts: torch.Tensor
    selected_weights: torch.Tensor
    mixture_output: torch.Tensor | None = None


@dataclass(slots=True)
class RouterInterpCaptureConfig:
    layers: tuple[int, ...]
    expert_labels: tuple[str, ...] = ()
    max_seq_len: int | None = None
    skip_first_token: bool = False
    artifact_dtype: str = "bfloat16"
    max_tokens: int | None = None


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
    capture_mixture_output: bool = False,
) -> tuple[RouterLayerCapture, ...]:
    """Capture router inputs, routing decisions, and optional MLP mixture outputs."""

    batch_size, sequence_length = input_batch_and_sequence_length(inputs)
    if batch_size is None or sequence_length is None:
        raise ValueError("RouterInterp capture requires batched `input_ids`.")
    raw_inputs: list[torch.Tensor] = []
    raw_outputs: list[Any] = []
    raw_mixture_outputs: list[Any] = []
    handles = []

    def hook(_module, hook_inputs, output):
        raw_inputs.append(hook_inputs[0].detach())
        raw_outputs.append(output)

    def mixture_hook(_module, _hook_inputs, output):
        raw_mixture_outputs.append(output.detach() if isinstance(output, torch.Tensor) else output)

    try:
        for layer in iter_flex_olmo_layers(model):
            handles.append(layer.mlp.gate.register_forward_hook(hook))
            if capture_mixture_output:
                handles.append(layer.mlp.register_forward_hook(mixture_hook))
        with torch.no_grad():
            model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    if not raw_inputs or len(raw_inputs) != len(raw_outputs):
        raise ValueError("No aligned FlexOlmo router inputs and outputs were captured.")
    if capture_mixture_output and len(raw_mixture_outputs) != len(raw_inputs):
        raise ValueError("No aligned FlexOlmo MLP mixture outputs were captured.")

    captures = []
    for layer_index, (router_input, output) in enumerate(zip(raw_inputs, raw_outputs)):
        router_scores = output[0] if isinstance(output, tuple) else output
        if not isinstance(router_scores, torch.Tensor):
            raise TypeError("FlexOlmo router did not return a tensor as its first gate output.")
        router_scores = _as_batch_sequence(router_scores, batch_size, sequence_length)
        probabilities = router_scores_to_probabilities(
            (router_scores,),
            scores_are_probabilities=None,
        )[0]
        router_input = _as_batch_sequence(router_input, batch_size, sequence_length).float()
        _require_finite("pre-router activations", router_input)
        _require_probability_rows("router probabilities", probabilities)
        effective_top_k = top_k or int(getattr(model.config, "num_experts_per_tok", 1))
        if not 1 <= effective_top_k <= probabilities.shape[-1]:
            raise ValueError(
                f"Router top-k={effective_top_k} is invalid for {probabilities.shape[-1]} experts."
            )
        selected_experts = probabilities.topk(k=effective_top_k, dim=-1).indices
        selected_weights = probabilities.gather(dim=-1, index=selected_experts)
        if isinstance(output, tuple) and len(output) >= 3:
            candidate_weights, candidate_indices = output[1], output[2]
            if isinstance(candidate_weights, torch.Tensor) and isinstance(candidate_indices, torch.Tensor):
                candidate_weights = _as_batch_sequence(candidate_weights, batch_size, sequence_length)
                candidate_indices = _as_batch_sequence(candidate_indices, batch_size, sequence_length).long()
                if candidate_indices.shape[-1] >= effective_top_k:
                    candidate_indices = candidate_indices[..., :effective_top_k]
                    candidate_weights = candidate_weights[..., :effective_top_k]
                    if (candidate_indices < 0).any() or (candidate_indices >= probabilities.shape[-1]).any():
                        raise ValueError("FlexOlmo gate returned out-of-range selected expert ids.")
                    # Retain the gate's actual selected order and mixture weights.
                    selected_experts = candidate_indices
                    selected_weights = candidate_weights
        mixture_output = None
        if capture_mixture_output:
            captured_output = raw_mixture_outputs[layer_index]
            if isinstance(captured_output, tuple):
                captured_output = captured_output[0]
            if not isinstance(captured_output, torch.Tensor):
                raise TypeError("FlexOlmo MLP did not return a tensor mixture output.")
            mixture_output = _as_batch_sequence(captured_output, batch_size, sequence_length).float()
            _require_finite("post-router mixture output", mixture_output)
        captures.append(
            RouterLayerCapture(
                router_input=router_input,
                router_scores=router_scores,
                router_probabilities=probabilities,
                selected_experts=selected_experts,
                selected_weights=selected_weights,
                mixture_output=mixture_output,
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
    captured_tokens = 0
    selected_layers = tuple(sorted(set(int(layer) for layer in config.layers)))

    for prompt_index, example in enumerate(examples):
        if config.max_tokens is not None and captured_tokens >= config.max_tokens:
            break
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
        # next-token lens, it has no reason to discard either boundary token.
        positions = list(range(start, token_count))
        if config.max_tokens is not None:
            remaining = config.max_tokens - captured_tokens
            positions = positions[: max(0, remaining)]
        if not positions:
            continue
        captures = capture_flexolmo_router_layers(
            model,
            {"input_ids": input_ids},
            capture_mixture_output=True,
        )
        if selected_layers and selected_layers[-1] >= len(captures):
            raise ValueError(f"Requested layer {selected_layers[-1]} but captured {len(captures)} layers.")
        artifact_layers = {}
        for layer in selected_layers:
            capture = captures[layer]
            router_width = int(capture.router_probabilities.shape[-1])
            if config.expert_labels and len(config.expert_labels) != router_width:
                raise ValueError(
                    "Configured expert labels do not match the router width: "
                    f"received {len(config.expert_labels)} labels for {router_width} experts."
                )
            artifact_layers[str(layer)] = {
                "router_input": capture.router_input[0, positions].cpu().to(artifact_dtype),
                "router_scores": capture.router_scores[0, positions].cpu().to(artifact_dtype),
                "router_probabilities": capture.router_probabilities[0, positions].cpu().to(artifact_dtype),
                "selected_experts": capture.selected_experts[0, positions].cpu(),
                "selected_weights": capture.selected_weights[0, positions].cpu().to(artifact_dtype),
                "mixture_output": (
                    capture.mixture_output[0, positions].cpu().to(artifact_dtype)
                    if capture.mixture_output is not None
                    else None
                ),
            }
        file_name = f"prompt_{prompt_index:06d}.pt"
        torch.save(
            {
                "prompt_index": prompt_index,
                "example_id": str(example.get("id", example.get("example_id", prompt_index))),
                "dataset_name": str(example.get("dataset_name", example.get("task", "unknown"))),
                "task": str(example.get("task", "unknown")),
                "domain": str(example.get("domain", "unknown")),
                "language": str(example.get("language", "unknown")),
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
                "dataset_name": str(example.get("dataset_name", example.get("task", "unknown"))),
                "task": str(example.get("task", "unknown")),
                "domain": str(example.get("domain", "unknown")),
                "language": str(example.get("language", "unknown")),
                "path": file_name,
                "num_tokens": len(positions),
            }
        )
        captured_tokens += len(positions)

    payload = {
        "format": "mi_lens.routerinterp.v1",
        "config": asdict(config),
        "num_prompts": len(manifest),
        "num_tokens": captured_tokens,
        "prompts": manifest,
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def selected_expert_targets(selected_experts: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Convert top-k ids into multi-label routing targets ``(token, expert)``."""

    if selected_experts.ndim != 2 or num_experts < 1:
        raise ValueError("Expected selected experts with shape (token, top_k).")
    if selected_experts.numel() == 0:
        raise ValueError("Selected experts cannot be empty.")
    if (selected_experts < 0).any() or (selected_experts >= num_experts).any():
        raise ValueError("Selected expert ids are outside the router expert range.")
    if (selected_experts.sort(dim=1).values[:, 1:] == selected_experts.sort(dim=1).values[:, :-1]).any():
        raise ValueError("Each token's routed expert ids must be unique.")
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
    _require_finite("feature activations", features)
    _require_finite("expert targets", expert_targets)
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
    result = torch.stack(scores)
    _require_finite("rho usefulness", result)
    return result


def top_rho_features(rho_scores: torch.Tensor, n_features: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the most routing-predictive SAE latents for each expert."""

    if rho_scores.ndim != 2 or rho_scores.shape[1] < 1 or n_features < 1:
        raise ValueError("rho scores must be non-empty (expert, feature) values.")
    _require_finite("rho scores", rho_scores)
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

    if features.ndim != 2 or expert_targets.ndim != 2 or features.shape[0] != expert_targets.shape[0]:
        raise ValueError("Routing probes require aligned non-empty feature and target matrices.")
    if features.shape[0] < 1 or steps < 1 or learning_rate <= 0:
        raise ValueError("Routing probe inputs, steps, and learning_rate must be positive.")
    _require_finite("routing-probe features", features)
    _require_finite("routing-probe targets", expert_targets)
    if (expert_targets < 0).any() or (expert_targets > 1).any():
        raise ValueError("Routing probe targets must be binary.")
    probe = RoutingProbe(features.shape[1], expert_targets.shape[1]).to(features.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=1e-4)
    for _ in range(steps):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(probe(features), expert_targets)
        if not torch.isfinite(loss):
            raise ValueError("Routing probe loss became NaN or infinity.")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return probe.eval()


def routing_recall(probe_logits: torch.Tensor, expert_targets: torch.Tensor, *, top_k: int) -> float:
    """Recall of actual experts among the probe's top-k predictions."""

    predicted = probe_logits.topk(k=top_k, dim=-1).indices
    predicted_targets = selected_expert_targets(predicted, expert_targets.shape[1])
    result = float((predicted_targets * expert_targets).sum(-1).div(top_k).mean().item())
    if not 0.0 <= result <= 1.0:
        raise ValueError("Routing recall is outside [0, 1].")
    return result


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
    result = float(
        (2 * true_positive / (2 * true_positive + false_positive + false_negative).clamp_min(1)).mean().item()
    )
    if not 0.0 <= result <= 1.0:
        raise ValueError("Macro-F1 is outside [0, 1].")
    return result


def sparse_by_magnitude(values: torch.Tensor, *, k: int) -> torch.Tensor:
    """Retain the ``k`` strongest signed coordinates of each representation."""

    if values.ndim != 2 or not 1 <= k <= values.shape[1]:
        raise ValueError("Values must be (token, feature) and `k` must be valid.")
    _require_finite("values to sparsify", values)
    indices = values.abs().topk(k=k, dim=-1).indices
    sparse = torch.zeros_like(values)
    return sparse.scatter(1, indices, values.gather(1, indices))


def feature_coactivation_ratio(features: torch.Tensor, feature_indices_by_expert: torch.Tensor) -> list[dict[str, float | int]]:
    """Paper-style feature co-activation relative to independence for each expert."""

    if features.ndim != 2 or feature_indices_by_expert.ndim != 2:
        raise ValueError("Feature coactivation requires feature and index matrices.")
    _require_finite("feature activations", features)
    if (feature_indices_by_expert < 0).any() or (feature_indices_by_expert >= features.shape[1]).any():
        raise ValueError("Feature coactivation indices are outside the feature range.")
    # Top-K SAE features are non-negative, while ITDA matching-pursuit
    # coefficients are signed. A feature is active when its magnitude is nonzero.
    active = features.abs() > 1e-12
    results = []
    for expert_index, indices in enumerate(feature_indices_by_expert):
        selected = active[:, indices]
        pair_ratios = []
        undefined_pairs = 0
        for left in range(selected.shape[1]):
            for right in range(left + 1, selected.shape[1]):
                observed = (selected[:, left] & selected[:, right]).float().mean()
                expected = selected[:, left].float().mean() * selected[:, right].float().mean()
                observed_value, expected_value = float(observed.item()), float(expected.item())
                if expected_value <= 1e-12:
                    # A ratio is mathematically undefined when both features
                    # never fire.  Exclude rather than emit a fake huge value.
                    if observed_value > 1e-12:
                        raise ValueError("Observed coactivation is positive despite zero independent expectation.")
                    undefined_pairs += 1
                    continue
                ratio = observed_value / expected_value
                if not torch.isfinite(torch.tensor(ratio)) or ratio < 0:
                    raise ValueError("Feature coactivation ratio is invalid.")
                pair_ratios.append((observed_value, expected_value, ratio))
        observed_values = [value[0] for value in pair_ratios]
        expected_values = [value[1] for value in pair_ratios]
        ratio_values = [value[2] for value in pair_ratios]
        results.append(
            {
                "expert": expert_index,
                "feature_count": int(selected.shape[1]),
                "pair_count": len(pair_ratios),
                "undefined_zero_expectation_pair_count": undefined_pairs,
                "mean_feature_firing_rate": float(selected.float().mean().item()),
                "mean_observed_coactivation": float(torch.tensor(observed_values).mean().item()) if observed_values else 0.0,
                "mean_independence_expectation": float(torch.tensor(expected_values).mean().item()) if expected_values else 0.0,
                "median_coactivation_ratio": float(torch.tensor(ratio_values).median().item()) if ratio_values else 1.0,
                "mean_coactivation_ratio": float(torch.tensor(ratio_values).mean().item()) if ratio_values else 1.0,
                "zero_coactivation_pair_fraction": (
                    float(sum(value == 0.0 for value in observed_values) / len(observed_values))
                    if observed_values
                    else 0.0
                ),
            }
        )
    return results


def feature_activation_diagnostics(features: torch.Tensor) -> dict[str, float | int]:
    """Summarise feature use on a fixed held-out token set."""

    if features.ndim != 2 or features.shape[0] < 1:
        raise ValueError("Expected non-empty (token, feature) activations.")
    _require_finite("feature activations", features)
    active = features.abs() > 1e-12
    firing_rates = active.float().mean(0)
    activation_means = features.mean(0)
    result = {
        "feature_count": int(features.shape[1]),
        "token_count": int(features.shape[0]),
        "dead_feature_count": int((firing_rates == 0).sum().item()),
        "dead_feature_fraction": float((firing_rates == 0).float().mean().item()),
        "mean_feature_firing_rate": float(firing_rates.mean().item()),
        "median_feature_firing_rate": float(firing_rates.median().item()),
        "p95_feature_firing_rate": float(torch.quantile(firing_rates, 0.95).item()),
        "mean_feature_activation": float(activation_means.mean().item()),
        "median_feature_activation": float(activation_means.median().item()),
    }
    for key in ("dead_feature_fraction", "mean_feature_firing_rate", "median_feature_firing_rate", "p95_feature_firing_rate"):
        if not 0.0 <= float(result[key]) <= 1.0:
            raise ValueError(f"{key} is outside [0, 1].")
    return result
