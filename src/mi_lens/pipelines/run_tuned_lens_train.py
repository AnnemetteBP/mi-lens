from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .common import (
    ModelLoadConfig,
    configure_hf_cache,
    metadata_payload,
    pipeline_project_paths,
    slugify,
    write_json,
)


def run_tuned_lens_train_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(config["project_root"])
    model_cfg = ModelLoadConfig(**config["model"])
    model_family = config["model_family"]
    model_label = config.get("model_label") or slugify(model_cfg.model_name)
    model_variant = config.get("model_variant")
    dataset_name = config.get("dataset_name")
    dataset_split_role = config.get("dataset_split_role")
    language = config.get("language")

    paths = pipeline_project_paths(project_root).for_family(model_family)
    for directory in (paths.tuned_lens_dir, paths.metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cache_dir = configure_hf_cache(project_root)
    output_path = project_root / config.get(
        "output_path",
        paths.tuned_lens_dir / f"{model_label}_{dataset_split_role or 'trainfit'}",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "tuned_lens",
        "train",
        "-o",
        str(output_path),
        "--model.name",
        model_cfg.model_name,
        "--data.name",
        str(project_root / config["examples_path"]),
        "--precision",
        str(config.get("precision", model_cfg.dtype)),
        "--max_seq_len",
        str(int(config.get("max_seq_len", 128))),
        "--tokens_per_step",
        str(int(config.get("tokens_per_step", 128))),
        "--num_steps",
        str(int(config.get("num_steps", 500))),
        "--per_gpu_batch_size",
        str(int(config.get("per_gpu_batch_size", 1))),
        "--loss",
        str(config.get("loss", "KL")),
        "--token_shift",
        str(int(config.get("token_shift", 0))),
        "--dataset_shuffle",
        "false",
        "--dataloader_shuffle",
        "false",
    ]

    optional_flags = {
        "checkpoint_freq": "--checkpoint_freq",
        "weight_decay": "--weight_decay",
        "lr_scale": "--lr_scale",
        "momentum": "--momentum",
        "warmup_steps": "--warmup_steps",
        "optimizer": "--optimizer",
    }
    for key, flag in optional_flags.items():
        if key in config and config[key] is not None:
            command.extend([flag, str(config[key])])

    if config.get("checkpoint_dir"):
        checkpoint_dir = project_root / config["checkpoint_dir"]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        command.extend(["--checkpoint_dir", str(checkpoint_dir)])

    if config.get("slow_tokenizer"):
        command.append("--slow_tokenizer")
    if config.get("tokenizer"):
        command.extend(["--tokenizer", str(config["tokenizer"])])
    if config.get("tokenizer_type"):
        command.extend(["--tokenizer_type", str(config["tokenizer_type"])])

    env = os.environ.copy()
    env["HF_HOME"] = str(cache_dir)
    env["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hub")
    env["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    pythonpath_parts = [
        str(project_root / "src"),
        str(project_root / "lenses" / "tuned_logit_lens"),
    ]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    subprocess.run(command, cwd=project_root, env=env, check=True)

    result = {
        "output_path": str(output_path),
        "command": command,
    }
    write_json(
        paths.metadata_dir
        / f"{model_label}_{dataset_split_role or 'trainfit'}_{language or 'all'}_tuned_lens_train.json",
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
                },
            ),
        },
    )
    return result
