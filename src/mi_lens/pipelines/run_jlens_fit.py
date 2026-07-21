from __future__ import annotations

from pathlib import Path
from typing import Any

import jlens

from .common import (
    ModelLoadConfig,
    clear_runtime_state,
    load_jsonl_records,
    load_model_and_tokenizer,
    metadata_payload,
    pipeline_project_paths,
    prompt_text_from_record,
    slugify,
    write_json,
)


def run_jlens_fit_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(config["project_root"])
    model_cfg = ModelLoadConfig(**config["model"])
    model_family = config["model_family"]
    model_label = config.get("model_label") or slugify(model_cfg.model_name)
    model_variant = config.get("model_variant")
    dataset_name = config.get("dataset_name")
    dataset_split_role = config.get("dataset_split_role")
    language = config.get("language")

    paths = pipeline_project_paths(project_root).for_family(model_family)
    for directory in (paths.jlens_dir, paths.metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    checkpoint_path = project_root / config.get(
        "checkpoint_path",
        paths.jlens_dir / f"{model_label}_{dataset_split_role or 'trainfit'}_ckpt.pt",
    )
    lens_path = project_root / config.get(
        "lens_path",
        paths.jlens_dir / f"{model_label}_{dataset_split_role or 'trainfit'}_lens.pt",
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    lens_path.parent.mkdir(parents=True, exist_ok=True)

    hf_model, tokenizer, cache_dir = load_model_and_tokenizer(
        model_cfg,
        project_root=project_root,
    )
    hf_model = hf_model.cpu()
    lens_model = jlens.from_hf(hf_model, tokenizer)

    records = load_jsonl_records(project_root / config["examples_path"])
    prompts = [prompt_text_from_record(record) for record in records]
    max_examples = config.get("max_examples")
    if max_examples is not None:
        prompts = prompts[: int(max_examples)]

    fit_kwargs: dict[str, Any] = {
        "max_seq_len": int(config.get("max_seq_len", 128)),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_every": int(config.get("checkpoint_every", 25)),
    }
    for key in (
        "source_layers",
        "target_layer",
        "dim_batch",
        "skip_first",
        "resume",
    ):
        if key in config and config[key] is not None:
            fit_kwargs[key] = config[key]

    lens = jlens.fit(
        lens_model,
        prompts=prompts,
        **fit_kwargs,
    )
    lens.save(str(lens_path))

    result = {
        "lens_path": str(lens_path),
        "checkpoint_path": str(checkpoint_path),
        "n_prompts": lens.n_prompts,
        "source_layers": lens.source_layers,
        "d_model": lens.d_model,
    }
    write_json(
        paths.metadata_dir
        / f"{model_label}_{dataset_split_role or 'trainfit'}_{language or 'all'}_jlens_fit.json",
        {
            "result": result,
            "metadata": metadata_payload(
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
                    "max_examples": len(prompts),
                    "fit_kwargs": fit_kwargs,
                },
            ),
        },
    )

    clear_runtime_state(lens, lens_model, hf_model, tokenizer)
    return result
