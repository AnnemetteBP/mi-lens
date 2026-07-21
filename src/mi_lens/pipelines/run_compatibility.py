from __future__ import annotations

from pathlib import Path
from typing import Any

import jlens
from tuned_lens import TunedLens

from ..methods import build_compatibility_report

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


def run_compatibility_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(config["project_root"])
    model_cfg = ModelLoadConfig(**config["model"])
    model_family = config["model_family"]
    model_label = config.get("model_label") or slugify(model_cfg.model_name)
    paths = pipeline_project_paths(project_root).for_family(model_family)
    for directory in (
        paths.compatibility_dir,
        paths.jlens_dir,
        paths.tuned_lens_dir,
        paths.metadata_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    hf_model, tokenizer, cache_dir = load_model_and_tokenizer(
        model_cfg,
        project_root=project_root,
    )
    hf_model = hf_model.cpu()

    prompt_records = load_jsonl_records(config["prompt_path"])
    prompts = [prompt_text_from_record(record) for record in prompt_records]

    tuned_lens = None
    if config.get("tuned_lens_path"):
        tuned_lens = TunedLens.from_model_and_pretrained(
            hf_model,
            str(project_root / config["tuned_lens_path"]),
        )
    elif config.get("test_tuned_from_model", True):
        tuned_lens = TunedLens.from_model(hf_model)

    report = build_compatibility_report(
        hf_model,
        tokenizer,
        prompts=prompts,
        model_label=model_label,
        tuned_lens=tuned_lens,
        jlens_lens=None,
        max_prompts=int(config.get("max_prompts", 3)),
        last_n_positions=int(config.get("last_n_positions", 2)),
        top_k=int(config.get("top_k", 5)),
    ).to_dict()

    fit_smoke = None
    if config.get("jlens_fit_smoke_prompt_path"):
        fit_records = load_jsonl_records(project_root / config["jlens_fit_smoke_prompt_path"])
        fit_prompts = [prompt_text_from_record(record) for record in fit_records]
        fit_prompts = fit_prompts[: int(config.get("jlens_fit_smoke_num_prompts", 8))]
        lens_model = jlens.from_hf(hf_model, tokenizer)
        checkpoint_path = paths.jlens_dir / f"{model_label}_fit_smoke_ckpt.pt"
        try:
            smoke_lens = jlens.fit(
                lens_model,
                prompts=fit_prompts,
                max_seq_len=int(config.get("max_seq_len", 128)),
                checkpoint_path=str(checkpoint_path),
                checkpoint_every=max(1, min(len(fit_prompts), 2)),
            )
            fit_smoke = {
                "ok": True,
                "checkpoint_path": str(checkpoint_path),
                "source_layers": smoke_lens.source_layers,
                "n_prompts": smoke_lens.n_prompts,
            }
        except Exception as exc:
            fit_smoke = {
                "ok": False,
                "checkpoint_path": str(checkpoint_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    output = {
        "compatibility_report": report,
        "jlens_fit_smoke": fit_smoke,
        "metadata": metadata_payload(
            model_config=model_cfg,
            cache_dir=cache_dir,
            extra={
                "prompt_path": str(config["prompt_path"]),
                "model_family": model_family,
                "model_label": model_label,
            },
        ),
    }
    write_json(
        paths.compatibility_dir / f"{model_label}_compatibility.json",
        output,
    )
    clear_runtime_state(tuned_lens, hf_model, tokenizer)
    return output
