"""Small, validated summaries for per-layer MoE routing distributions."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from mi_lens.adapters.router_outputs import router_scores_to_probabilities


def summarize_router_probabilities(
    router_probabilities: Sequence[torch.Tensor],
    *,
    top_k: int = 2,
) -> list[dict[str, float | int | list[float]]]:
    """Return compact, per-layer routing summaries without retaining full tensors."""

    if top_k < 1:
        raise ValueError(f"`top_k` must be positive, received {top_k}.")

    summaries: list[dict[str, float | int | list[float]]] = []
    for layer_index, layer_probs in enumerate(router_probabilities):
        if layer_probs.ndim != 3:
            raise ValueError(
                "Router probabilities must have shape (batch, token, expert), "
                f"received {tuple(layer_probs.shape)} for layer {layer_index}."
            )
        num_experts = int(layer_probs.shape[-1])
        if top_k > num_experts:
            raise ValueError(
                f"`top_k`={top_k} exceeds {num_experts} experts in layer {layer_index}."
            )

        probs = layer_probs.float()
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
        top_values, top_indices = probs.topk(k=top_k, dim=-1)
        top1_assignments = top_indices[..., 0].reshape(-1)
        usage = torch.bincount(top1_assignments, minlength=num_experts).float()
        usage = usage / usage.sum().clamp_min(1)

        summaries.append(
            {
                "layer": layer_index,
                "num_experts": num_experts,
                "num_tokens": int(probs.shape[0] * probs.shape[1]),
                "mean_entropy": float(entropy.mean().item()),
                "normalized_mean_entropy": float(
                    (entropy.mean() / torch.log(torch.tensor(float(num_experts)))).item()
                )
                if num_experts > 1
                else 0.0,
                "mean_top1_prob": float(top_values[..., 0].mean().item()),
                "mean_topk_prob_mass": float(top_values.sum(dim=-1).mean().item()),
                "mean_top1_top2_margin": float(
                    (top_values[..., 0] - top_values[..., 1]).mean().item()
                )
                if top_k > 1
                else 0.0,
                "top1_expert_usage": usage.cpu().tolist(),
            }
        )
    return summaries


def summarize_router_scores(
    router_scores: Sequence[torch.Tensor],
    *,
    top_k: int = 2,
    scores_are_probabilities: bool = False,
) -> list[dict[str, float | int | list[float]]]:
    """Convert router scores to probabilities and return per-layer summaries."""

    probabilities = router_scores_to_probabilities(
        router_scores,
        scores_are_probabilities=scores_are_probabilities,
    )
    return summarize_router_probabilities(probabilities, top_k=top_k)


def _topk_indicator(layer_probabilities: torch.Tensor, top_k: int) -> torch.Tensor:
    """Return a token-by-expert binary activation matrix for one router layer."""

    if layer_probabilities.ndim != 3:
        raise ValueError(
            "Router probabilities must have shape (batch, token, expert), "
            f"received {tuple(layer_probabilities.shape)}."
        )
    num_experts = layer_probabilities.shape[-1]
    if not 1 <= top_k <= num_experts:
        raise ValueError(
            f"`top_k` must be in [1, {num_experts}], received {top_k}."
        )

    selected = layer_probabilities.topk(k=top_k, dim=-1).indices.reshape(-1, top_k)
    indicator = torch.zeros(
        (selected.shape[0], num_experts),
        dtype=torch.float32,
        device=layer_probabilities.device,
    )
    indicator.scatter_(1, selected, 1.0)
    return indicator


def _fixed_k_weighted_null(
    activation_counts: torch.Tensor,
    *,
    num_tokens: int,
    top_k: int,
    num_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate random fixed-k co-activation while retaining expert prevalence.

    Each token samples ``top_k`` unique experts using the observed marginal
    activation frequencies as weights. This is deliberately a *routing-null*,
    not a semantic null: it tells us whether a pair occurs more often than
    expected from expert prevalence and the fixed top-k constraint alone.
    """

    num_experts = int(activation_counts.numel())
    if top_k == 1 or num_samples < 1:
        zeros = torch.zeros(
            (num_experts, num_experts), dtype=torch.float32, device="cpu"
        )
        return zeros, zeros

    weights = activation_counts.detach().float().cpu().clamp_min(1e-6)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples = []
    for _ in range(num_samples):
        selected = torch.multinomial(
            weights.expand(num_tokens, -1),
            num_samples=top_k,
            replacement=False,
            generator=generator,
        )
        indicator = torch.zeros(
            (num_tokens, num_experts), dtype=torch.float32, device="cpu"
        )
        indicator.scatter_(1, selected, 1.0)
        samples.append(indicator.T @ indicator)

    stacked = torch.stack(samples)
    return stacked.mean(dim=0), stacked.std(dim=0, correction=0)


def summarize_router_coactivation(
    router_probabilities: Sequence[torch.Tensor],
    *,
    top_k: int = 2,
    null_samples: int = 128,
    seed: int = 0,
) -> list[dict[str, int | float | list[list[float]] | list[float]]]:
    """Summarize expert-pair co-activation with a prevalence-aware null.

    Pass aggregated batches rather than a prompt at a time. The result contains
    compact matrices that can be written to JSON and plotted later, while raw
    router tensors remain optional.
    """

    if null_samples < 0:
        raise ValueError(f"`null_samples` must be non-negative, received {null_samples}.")

    summaries = []
    for layer_index, layer_probs in enumerate(router_probabilities):
        indicator = _topk_indicator(layer_probs.float(), top_k)
        num_tokens, num_experts = indicator.shape
        activation_counts = indicator.sum(dim=0)
        coactivation_counts = indicator.T @ indicator

        if not torch.equal(coactivation_counts, coactivation_counts.T):
            raise AssertionError("Co-activation counts must be symmetric.")
        if not torch.equal(coactivation_counts.diagonal(), activation_counts):
            raise AssertionError("Co-activation diagonal must equal expert activation counts.")

        conditional = coactivation_counts / activation_counts[:, None].clamp_min(1)
        marginal = activation_counts / float(num_tokens)
        independent_expected = marginal[:, None] * marginal[None, :] * float(num_tokens)
        lift = coactivation_counts / independent_expected.clamp_min(1e-12)
        lift.fill_diagonal_(1.0)

        null_mean, null_std = _fixed_k_weighted_null(
            activation_counts,
            num_tokens=num_tokens,
            top_k=top_k,
            num_samples=null_samples,
            seed=seed + layer_index,
        )
        observed_cpu = coactivation_counts.detach().float().cpu()
        null_z = (observed_cpu - null_mean) / null_std.clamp_min(1e-6)
        null_z.fill_diagonal_(0.0)

        summaries.append(
            {
                "layer": layer_index,
                "num_tokens": int(num_tokens),
                "num_experts": int(num_experts),
                "top_k": int(top_k),
                "null_samples": int(null_samples),
                "expert_activation_rate": marginal.detach().cpu().tolist(),
                "coactivation_count": observed_cpu.tolist(),
                "conditional_coactivation": conditional.detach().cpu().tolist(),
                "coactivation_lift": lift.detach().cpu().tolist(),
                "null_mean_coactivation": null_mean.tolist(),
                "null_std_coactivation": null_std.tolist(),
                "null_z_score": null_z.tolist(),
            }
        )
    return summaries


def summarize_router_score_coactivation(
    router_scores: Sequence[torch.Tensor],
    *,
    top_k: int = 2,
    scores_are_probabilities: bool = False,
    null_samples: int = 128,
    seed: int = 0,
) -> list[dict[str, int | float | list[list[float]] | list[float]]]:
    """Convert router scores then summarize co-activation."""

    probabilities = router_scores_to_probabilities(
        router_scores,
        scores_are_probabilities=scores_are_probabilities,
    )
    return summarize_router_coactivation(
        probabilities,
        top_k=top_k,
        null_samples=null_samples,
        seed=seed,
    )
