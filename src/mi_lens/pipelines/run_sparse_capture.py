"""Pipeline for reusable residual activations used by SAE-style methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mi_lens.sparse import ResidualCaptureConfig, capture_residual_activation_shards

from .common import (
    ModelLoadConfig,
    clear_runtime_state,
    load_jsonl_records,
    load_model_and_tokenizer,
    metadata_payload,
    pipeline_project_paths,
    slugify,
    write_json,
)


def run_sparse_capture_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Load a model once and write a reusable, sharded residual dataset."""

    project_root = Path(config["project_root"])
    model_config = ModelLoadConfig(**config["model"])
    model_family = str(config["model_family"])
    model_label = str(config.get("model_label") or slugify(model_config.model_name))
    dataset_name = str(config.get("dataset_name", "dataset"))
    split_role = str(config.get("dataset_split_role", "data"))
    language = str(config.get("language", "all"))
    paths = pipeline_project_paths(project_root).for_family(model_family)

    output_dir = project_root / config.get(
        "output_path",
        paths.family_dir
        / "sparse"
        / "activations"
        / model_label
        / f"{dataset_name}_{split_role}_{language}",
    )
    records = load_jsonl_records(project_root / config["examples_path"])
    max_examples = config.get("max_examples")
    if max_examples is not None:
        records = records[: int(max_examples)]

    capture_config = ResidualCaptureConfig(
        layers=tuple(int(layer) for layer in config["layers"]),
        max_seq_len=int(config.get("max_seq_len", 512)),
        start_token_idx=int(config.get("start_token_idx", 0)),
        end_token_idx=config.get("end_token_idx"),
        skip_first_token=bool(config.get("skip_first_token", True)),
        shard_size_tokens=int(config.get("shard_size_tokens", 8192)),
        artifact_dtype=str(config.get("artifact_dtype", "bfloat16")),
    )

    model, tokenizer, cache_dir = load_model_and_tokenizer(
        model_config,
        project_root=project_root,
    )
    try:
        manifest = capture_residual_activation_shards(
            model,
            tokenizer,
            records,
            output_dir=output_dir,
            config=capture_config,
            metadata={
                "model_family": model_family,
                "model_label": model_label,
                "model_variant": config.get("model_variant"),
                "dataset_name": dataset_name,
                "dataset_split_role": split_role,
                "language": language,
                "examples_path": str(config["examples_path"]),
            },
        )
    finally:
        clear_runtime_state(model, tokenizer)

    metadata_path = output_dir / "capture_metadata.json"
    write_json(
        metadata_path,
        {
            "metadata": metadata_payload(
                model_config=model_config,
                cache_dir=cache_dir,
                extra={"output_dir": str(output_dir), "num_examples": len(records)},
            ),
            "manifest_path": str(output_dir / "manifest.json"),
        },
    )
    return {
        "output_dir": str(output_dir),
        "manifest_path": str(output_dir / "manifest.json"),
        "metadata_path": str(metadata_path),
        "num_prompts": manifest["num_prompts"],
        "num_tokens": manifest["num_tokens"],
        "layers": manifest["layers"],
    }
