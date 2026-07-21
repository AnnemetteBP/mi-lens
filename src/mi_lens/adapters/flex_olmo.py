from __future__ import annotations

from typing import Iterable

import torch

from .base import BaseAdapter
from .router_outputs import (
    input_batch_and_sequence_length,
    normalize_router_layers,
    router_scores_to_probabilities,
)


def iter_flex_olmo_layers(model) -> Iterable[torch.nn.Module]:
    """
    Yield decoder layers from either a bare FlexOlmoModel or FlexOlmoForCausalLM.
    """

    base_model = getattr(model, "model", model)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        raise ValueError("Model does not expose FlexOlmo layers via `.layers` or `.model.layers`.")
    return layers


class FlexOlmoAdapter(BaseAdapter):
    """
    Capture router outputs from local FlexOlmo/FlexMoRE-style models.

    Flex checkpoints expose either normalized routing probabilities or raw router
    logits as their first gate output. The shared converter handles both forms.
    """

    def _collect_router_probs(self, model, inputs):
        router_probs = []
        handles = []

        def hook_fn(_module, _args, output):
            probs = output[0] if isinstance(output, tuple) else output
            router_probs.append(probs.detach())

        try:
            for layer in iter_flex_olmo_layers(model):
                handles.append(layer.mlp.gate.register_forward_hook(hook_fn))

            with torch.no_grad():
                model(**inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        if not router_probs:
            raise ValueError("No router outputs were captured from the FlexOlmo model.")

        batch_size, sequence_length = input_batch_and_sequence_length(inputs)
        return normalize_router_layers(
            router_probs,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )

    def get_router_logits(self, model, inputs):
        router_outputs = self._collect_router_probs(model, inputs)
        router_probs = router_scores_to_probabilities(
            router_outputs,
            scores_are_probabilities=None,
        )
        return tuple(torch.log(layer_probs.clamp_min(1e-9)) for layer_probs in router_probs)

    def get_router_probs(self, model, inputs):
        return router_scores_to_probabilities(
            self._collect_router_probs(model, inputs),
            scores_are_probabilities=None,
        )

    def router_logits_to_probs(self, router_logits):
        return router_scores_to_probabilities(router_logits)
