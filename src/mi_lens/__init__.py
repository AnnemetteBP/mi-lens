"""Methods and pipelines for analysing intermediate LLM representations."""

# Keep the package root dependency-light. Optional model-family adapters are
# loaded only when a workflow explicitly selects them.

__all__ = [
    "adapters",
    "methods",
    "pipelines",
    "plotting",
    "schemas",
    "sparse",
]
