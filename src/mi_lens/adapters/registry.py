"""Lazy registry for optional architecture adapters."""

from importlib import import_module


ADAPTER_REGISTRY = {
    "flex_olmo": ("mi_lens.adapters.flex_olmo", "FlexOlmoAdapter"),
    "hf_moe": ("mi_lens.adapters.hf_moe", "HFMoEAdapter"),
}


def get_adapter(name: str):
    """Return an adapter class, importing optional dependencies on demand."""

    try:
        module_name, class_name = ADAPTER_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown adapter {name!r}.") from exc

    module = import_module(module_name)
    return getattr(module, class_name)
