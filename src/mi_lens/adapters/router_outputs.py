"""Normalize router outputs across Hugging Face and FlexOlmo-style MoE models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def input_batch_and_sequence_length(inputs: Mapping[str, Any]) -> tuple[int | None, int | None]:
    """Infer batch and sequence dimensions from standard model inputs."""

    input_ids = inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim < 2:
        return None, None
    return int(input_ids.shape[0]), int(input_ids.shape[1])


def normalize_router_layer(
    scores: torch.Tensor,
    *,
    batch_size: int | None = None,
    sequence_length: int | None = None,
) -> torch.Tensor:
    """Return one layer's router scores with shape ``(batch, token, expert)``.

    Hugging Face MoE models commonly expose flattened ``(batch * token, expert)``
    router scores, while FlexOlmo hooks may expose either that form or the already
    shaped three-dimensional tensor. A flat tensor is reshaped when the input
    dimensions are known; otherwise it remains a single synthetic batch.
    """

    if not isinstance(scores, torch.Tensor):
        raise TypeError(f"Router scores must be a torch.Tensor, received {type(scores)!r}.")
    if scores.ndim == 3:
        return scores
    if scores.ndim != 2:
        raise ValueError(
            "Router scores must have shape (batch, token, expert) or "
            f"(batch * token, expert), received {tuple(scores.shape)}."
        )

    if batch_size is not None and sequence_length is not None:
        expected_rows = batch_size * sequence_length
        if scores.shape[0] == expected_rows:
            return scores.reshape(batch_size, sequence_length, scores.shape[-1])
    return scores.unsqueeze(0)


def normalize_router_layers(
    router_outputs: torch.Tensor | Sequence[torch.Tensor],
    *,
    batch_size: int | None = None,
    sequence_length: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """Normalize per-layer router outputs to a non-empty tuple of 3D tensors."""

    if isinstance(router_outputs, torch.Tensor):
        if router_outputs.ndim == 4:
            raw_layers = tuple(router_outputs.unbind(dim=0))
        else:
            raw_layers = (router_outputs,)
    elif isinstance(router_outputs, Sequence) and not isinstance(router_outputs, (str, bytes)):
        raw_layers = tuple(router_outputs)
    else:
        raise TypeError(
            "Router outputs must be a tensor or a sequence of per-layer tensors, "
            f"received {type(router_outputs)!r}."
        )

    if not raw_layers:
        raise ValueError("Router outputs must contain at least one layer.")
    return tuple(
        normalize_router_layer(
            layer_scores,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        for layer_scores in raw_layers
    )


def router_scores_to_probabilities(
    router_scores: Sequence[torch.Tensor],
    *,
    scores_are_probabilities: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Convert normalized router scores to validated probability distributions."""

    probabilities = []
    for layer_scores in router_scores:
        if scores_are_probabilities:
            layer_probs = layer_scores.float()
        else:
            layer_probs = torch.softmax(layer_scores.float(), dim=-1)

        if not torch.isfinite(layer_probs).all():
            raise ValueError("Router probabilities contain non-finite values.")
        if (layer_probs < 0).any():
            raise ValueError("Router probabilities contain negative values.")
        if not torch.allclose(
            layer_probs.sum(dim=-1),
            torch.ones_like(layer_probs[..., 0]),
            rtol=1e-4,
            atol=2e-3,
        ):
            raise ValueError("Router probabilities must sum to one over experts.")
        probabilities.append(layer_probs)
    return tuple(probabilities)
