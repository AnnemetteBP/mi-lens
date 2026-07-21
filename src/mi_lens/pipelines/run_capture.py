from __future__ import annotations

from pathlib import Path
from typing import Any

import jlens
from tuned_lens import TunedLens

from ..methods import (
    BatchCaptureConfig,
    capture_lens_batch,
    summarize_capture_rows,
    write_capture_rows_jsonl,
)

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


def _summary_groups() -> list[list[str]]:
    return [
        ["model_label", "model_variant", "lens_name", "language", "layer"],
        ["model_family", "model_variant", "lens_name", "language"],
        ["model_family", "lens_name", "layer"],
    ]


def run_capture_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(config["project_root"])
    model_cfg = ModelLoadConfig(**config["model"])
    model_family = config["model_family"]
    model_label = config.get("model_label") or slugify(model_cfg.model_name)
    model_variant = config.get("model_variant")
    dataset_name = config.get("dataset_name")
    dataset_split_role = config.get("dataset_split_role")
    language = config.get("language")

    family_paths = pipeline_project_paths(project_root).for_family(model_family)
    for directory in (
        family_paths.capture_rows_dir,
        family_paths.capture_tensors_dir,
        family_paths.summaries_dir,
        family_paths.metadata_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    hf_model, tokenizer, cache_dir = load_model_and_tokenizer(
        model_cfg,
        project_root=project_root,
    )
    hf_model = hf_model.cpu()

    lens_model = jlens.from_hf(hf_model, tokenizer)
    tuned_lens = None
    if config.get("tuned_lens_path"):
        tuned_lens = TunedLens.from_model_and_pretrained(
            hf_model,
            str(project_root / config["tuned_lens_path"]),
        )

    jlens_lens = None
    if config.get("jlens_lens_path"):
        jlens_lens = jlens.JacobianLens.load(str(project_root / config["jlens_lens_path"]))

    examples = load_jsonl_records(project_root / config["examples_path"])
    max_examples = config.get("max_examples")
    if max_examples is not None:
        examples = examples[: int(max_examples)]

    capture_cfg = BatchCaptureConfig(
        max_seq_len=int(config.get("max_seq_len", 512)),
        layer_stride=int(config.get("layer_stride", 1)),
        last_n_positions=int(config.get("last_n_positions", 12)),
        start_token_idx=config.get("start_token_idx"),
        end_token_idx=config.get("end_token_idx"),
        top_k=int(config.get("top_k", 5)),
        surface_top_k=int(config.get("surface_top_k", 5)),
        include_topk_tokens=bool(config.get("include_topk_tokens", True)),
        include_surface_metrics=bool(config.get("include_surface_metrics", True)),
        save_prompt_artifacts=bool(config.get("save_prompt_artifacts", True)),
        prompt_artifact_dir=family_paths.capture_tensors_dir,
        artifact_dtype=str(config.get("artifact_dtype", "float32")),
    )

    rows = capture_lens_batch(
        hf_model,
        tokenizer,
        examples=examples,
        model_label=model_label,
        model_variant=model_variant,
        model_family=model_family,
        dataset_name=dataset_name,
        dataset_split_role=dataset_split_role,
        language=language,
        lens_model=lens_model,
        jlens_lens=jlens_lens,
        tuned_lens=tuned_lens,
        lens_names=tuple(config.get("lens_names", ["logit", "tuned", "jlens"])),
        layers=config.get("layers"),
        config=capture_cfg,
    )

    row_file = (
        family_paths.capture_rows_dir
        / f"{model_label}_{dataset_split_role or 'data'}_{language or 'all'}.jsonl"
    )
    write_capture_rows_jsonl(rows, row_file)

    summaries = {}
    for group in _summary_groups():
        key = "__".join(group)
        summaries[key] = summarize_capture_rows(rows, group_by=group)
        write_json(
            family_paths.summaries_dir
            / f"{model_label}_{dataset_split_role or 'data'}_{language or 'all'}__{key}.json",
            {"group_by": group, "rows": summaries[key]},
        )

    metadata = metadata_payload(
        model_config=model_cfg,
        cache_dir=cache_dir,
        extra={
            "model_family": model_family,
            "model_label": model_label,
            "model_variant": model_variant,
            "dataset_name": dataset_name,
            "dataset_split_role": dataset_split_role,
            "language": language,
            "examples_path": str(config["examples_path"]),
            "row_file": str(row_file),
            "prompt_artifact_dir": str(family_paths.capture_tensors_dir),
            "num_examples": len(examples),
        },
    )
    write_json(
        family_paths.metadata_dir
        / f"{model_label}_{dataset_split_role or 'data'}_{language or 'all'}_capture_metadata.json",
        metadata,
    )

    clear_runtime_state(jlens_lens, tuned_lens, lens_model, hf_model, tokenizer)
    return {
        "row_file": str(row_file),
        "num_rows": len(rows),
        "summary_groups": list(summaries.keys()),
    }
