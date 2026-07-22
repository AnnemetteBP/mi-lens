"""Router-aware, output-facing measurements for FlexLens.

FlexLens is deliberately a measurement layer rather than a trained sparse
dictionary. It connects router choices and residual readouts to target-token
behaviour while keeping full vocabulary logits out of persisted rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class FlexLensConfig:
    """Numerical settings shared by FlexLens token-level measurements."""

    top_k: int = 8
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("FlexLens `top_k` must be positive.")
        if not 0.0 < self.eps < 1.0:
            raise ValueError("FlexLens `eps` must lie strictly between zero and one.")


@dataclass(frozen=True, slots=True)
class FlexLensTokenMetrics:
    """Compact output-facing metrics for one or more token positions."""

    target_token_id: int
    top1_token_id: int
    target_probability: float
    target_surprisal: float
    target_rank: int
    top1_probability: float
    entropy: float
    margin_top1_top2: float
    top_k_token_ids: tuple[int, ...]


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinity.")


def _as_float_logits(logits: torch.Tensor) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor) or logits.ndim < 1:
        raise ValueError("Logits must be a tensor with a vocabulary dimension.")
    logits = logits.float()
    _require_finite("FlexLens logits", logits)
    return logits


def _validate_target_ids(target_token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    target_token_ids = target_token_ids.long()
    if (target_token_ids < 0).any() or (target_token_ids >= vocab_size).any():
        raise ValueError("Target token IDs must lie within the logits vocabulary.")
    return target_token_ids


def softmax_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Return validated probabilities from logits along the final dimension."""

    logits = _as_float_logits(logits)
    probabilities = torch.softmax(logits, dim=-1)
    _require_finite("FlexLens probabilities", probabilities)
    if (probabilities < 0).any():
        raise ValueError("Softmax produced a negative probability.")
    if not torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones_like(probabilities[..., 0]),
        rtol=2e-5,
        atol=2e-6,
    ):
        raise ValueError("FlexLens probabilities are not normalised.")
    return probabilities


def readout_hidden_states(
    hidden_states: torch.Tensor,
    output_projection: torch.Tensor,
    *,
    output_bias: torch.Tensor | None = None,
    final_normalization: Any | None = None,
) -> torch.Tensor:
    """Project hidden states through a model's output readout.

    ``hidden_states`` may be shaped ``(..., d_model)``. The projection must be
    shaped ``(vocab_size, d_model)``. A model's final normalization can be
    supplied explicitly; omitting it is useful for controlled residual
    comparisons but should be recorded in the output metadata.
    """

    if hidden_states.ndim < 1 or output_projection.ndim != 2:
        raise ValueError("Hidden states and output projection have invalid ranks.")
    if hidden_states.shape[-1] != output_projection.shape[-1]:
        raise ValueError("Hidden-state width does not match output projection width.")
    states = hidden_states.float()
    projection = output_projection.float()
    if final_normalization is not None:
        states = final_normalization(states)
    logits = torch.matmul(states, projection.transpose(0, 1))
    if output_bias is not None:
        if output_bias.ndim != 1 or output_bias.shape[0] != projection.shape[0]:
            raise ValueError("Output bias must match the projection vocabulary size.")
        logits = logits + output_bias.float()
    _require_finite("FlexLens readout logits", logits)
    return logits


def token_metrics(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor | int,
    *,
    config: FlexLensConfig | None = None,
) -> FlexLensTokenMetrics | list[FlexLensTokenMetrics]:
    """Compute compact target-token and top-k metrics without saving logits."""

    cfg = config or FlexLensConfig()
    logits = _as_float_logits(logits)
    if logits.ndim != 1:
        raise ValueError("`token_metrics` expects one vocabulary logit vector.")
    target_ids = torch.as_tensor(target_token_ids, device=logits.device)
    target_ids = _validate_target_ids(target_ids, logits.shape[-1])
    if target_ids.numel() != 1:
        raise ValueError("`token_metrics` accepts exactly one target token ID.")
    target_id = int(target_ids.item())
    probabilities = softmax_probabilities(logits)
    top_k = min(cfg.top_k, logits.shape[-1])
    top_values, top_ids = torch.topk(logits, k=top_k, dim=-1)
    target_probability = probabilities[target_id]
    target_rank = int((logits > logits[target_id]).sum().item()) + 1
    entropy = -(probabilities * probabilities.clamp_min(cfg.eps).log()).sum()
    margin = top_values[0] - top_values[1] if top_k > 1 else torch.tensor(0.0)
    result = FlexLensTokenMetrics(
        target_token_id=target_id,
        top1_token_id=int(top_ids[0].item()),
        target_probability=float(target_probability.item()),
        target_surprisal=float(-target_probability.clamp_min(cfg.eps).log().item()),
        target_rank=target_rank,
        top1_probability=float(probabilities[top_ids[0]].item()),
        entropy=float(entropy.item()),
        margin_top1_top2=float(margin.item()),
        top_k_token_ids=tuple(int(token_id) for token_id in top_ids.tolist()),
    )
    return result


def batch_token_metrics(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    *,
    config: FlexLensConfig | None = None,
) -> list[FlexLensTokenMetrics]:
    """Apply :func:`token_metrics` to a ``(tokens, vocabulary)`` tensor."""

    if logits.ndim != 2 or target_token_ids.ndim != 1:
        raise ValueError("Batch logits must be 2-D and targets must be 1-D.")
    if logits.shape[0] != target_token_ids.shape[0]:
        raise ValueError("Batch logits and target counts must match.")
    return [
        token_metrics(row, target, config=config)
        for row, target in zip(logits, target_token_ids, strict=True)
    ]


def router_metrics(
    router_probabilities: torch.Tensor,
    *,
    config: FlexLensConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Compute validated router entropy, margins, and expert ranks.

    The returned tensors retain the leading token dimensions and are suitable
    for streaming row creation. No expert identity is inferred from labels.
    """

    cfg = config or FlexLensConfig()
    probabilities = router_probabilities.float()
    if probabilities.ndim < 1 or probabilities.shape[-1] < 1:
        raise ValueError("Router probabilities must have an expert dimension.")
    _require_finite("FlexLens router probabilities", probabilities)
    if (probabilities < -cfg.eps).any() or (probabilities > 1.0 + cfg.eps).any():
        raise ValueError("Router probabilities must lie in [0, 1].")
    row_sums = probabilities.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), rtol=2e-4, atol=2e-5):
        raise ValueError("Router probability rows are not normalised.")
    top_k = min(2, probabilities.shape[-1])
    top_values, top_ids = torch.topk(probabilities, k=top_k, dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(cfg.eps).log()).sum(dim=-1)
    return {
        "entropy": entropy,
        "top1_expert": top_ids[..., 0],
        "top1_probability": top_values[..., 0],
        "top2_probability": top_values[..., 1] if top_k == 2 else torch.zeros_like(top_values[..., 0]),
        "top1_top2_margin": (
            top_values[..., 0] - top_values[..., 1]
            if top_k == 2
            else top_values[..., 0]
        ),
        "expert_rank": probabilities.argsort(dim=-1, descending=True),
    }


def target_logit_contributions(
    expert_outputs: torch.Tensor,
    router_probabilities: torch.Tensor,
    output_projection: torch.Tensor,
    target_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Project weighted expert outputs onto target-token logits.

    Inputs use shapes ``(..., experts, d_model)``, ``(..., experts)``,
    ``(vocab_size, d_model)``, and ``(...)``. Only target-token contributions
    are returned, avoiding a ``tokens x experts x vocabulary`` allocation.
    """

    if expert_outputs.ndim < 2 or router_probabilities.shape != expert_outputs.shape[:-1]:
        raise ValueError("Expert outputs and router probabilities are misaligned.")
    if output_projection.ndim != 2 or output_projection.shape[-1] != expert_outputs.shape[-1]:
        raise ValueError("Output projection does not match expert output width.")
    target_token_ids = _validate_target_ids(
        torch.as_tensor(target_token_ids, device=expert_outputs.device),
        output_projection.shape[0],
    )
    if target_token_ids.shape != expert_outputs.shape[:-2]:
        raise ValueError("Target token IDs must match the leading expert-output dimensions.")
    selected_output = output_projection[target_token_ids]
    contributions = (expert_outputs.float() * selected_output.unsqueeze(-2)).sum(dim=-1)
    contributions = contributions * router_probabilities.float()
    _require_finite("FlexLens target logit contributions", contributions)
    return contributions
