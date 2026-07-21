from .base import BaseAdapter
from .router_outputs import (
    input_batch_and_sequence_length,
    normalize_router_layers,
    router_scores_to_probabilities,
)


class HFMoEAdapter(BaseAdapter):
    def get_router_logits(self, model, inputs):
        outputs = model(**inputs, output_router_logits=True, use_cache=False)
        raw_logits = getattr(outputs, "router_logits", None)
        if raw_logits is None:
            raise ValueError(
                "The model did not return `router_logits`. It may not support "
                "`output_router_logits=True` through the Hugging Face API."
            )
        batch_size, sequence_length = input_batch_and_sequence_length(inputs)
        return normalize_router_layers(
            raw_logits,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )

    def get_router_probs(self, model, inputs):
        return self.router_logits_to_probs(self.get_router_logits(model, inputs))

    def router_logits_to_probs(self, router_logits):
        return router_scores_to_probabilities(router_logits)
