"""Portable resolution of locally hosted research checkpoints."""

from __future__ import annotations

import os
from pathlib import Path


FLEXMORE_UCLOUD_MODEL_ROOT = Path("/work/training/FlexMoRE/models")
FLEXMORE_MODEL_ROOT_ENV = "MI_LENS_FLEXMORE_MODEL_ROOT"


def resolve_flexmore_checkpoint(
    checkpoint_name: str,
    *,
    model_root: str | Path | None = None,
) -> Path:
    """Resolve a named FlexMoRE checkpoint on UCloud or an override mount."""

    checkpoint = Path(checkpoint_name)
    if checkpoint.name != checkpoint_name:
        raise ValueError(
            "`checkpoint_name` must be a checkpoint directory name, not a path."
        )

    root = Path(
        model_root
        or os.environ.get(FLEXMORE_MODEL_ROOT_ENV)
        or FLEXMORE_UCLOUD_MODEL_ROOT
    )
    return root / checkpoint
