#!/usr/bin/env python3
"""Build and optionally run the complete FlexOlmo/FlexDanish RouterInterp batch."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Flex RouterInterp matrix JSON.")
    parser.add_argument("--run", action="store_true", help="Run the generated resumable batch immediately.")
    args = parser.parse_args()

    matrix_path = Path(args.config).resolve()
    matrix = _load_json(matrix_path)
    project_value = Path(str(matrix.get("project_root", ROOT)))
    project_root = project_value if project_value.is_absolute() else (matrix_path.parent.parent.parent / project_value)
    project_root = project_root.resolve()
    batch_name = str(matrix["batch_name"])
    source = dict(matrix["source"])
    capture = dict(matrix["capture"])
    analysis = dict(matrix["analysis"])
    models = list(matrix["models"])
    generated_dir = project_root / "tmp" / "routerinterp" / "batch_configs" / batch_name
    jobs = []

    for model in models:
        label = str(model["label"])
        checkpoint = str(model["checkpoint"])
        expert_labels = [str(label) for label in model.get("expert_labels", ())]
        model_config = {
            "model_name": checkpoint,
            "flexmore_checkpoint": checkpoint,
            "dtype": "bfloat16",
            "device": "cuda",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "tokenizer_class": "GPT2Tokenizer",
        }
        base_output = f"tmp/routerinterp/flex/{label}"
        sae_fit_config = {
            "project_root": str(project_root),
            "model": model_config,
            "output_path": f"{base_output}/streamed_sae",
            "expert_labels": expert_labels,
            **capture,
            **analysis,
        }
        if "dataset_config_path" in source:
            sae_fit_config.update({"dataset_config_path": source["dataset_config_path"], "dataset_role": "sae_fit"})
        else:
            sae_fit_config["examples_glob"] = source["sae_fit_glob"]
        train_config = {
            "project_root": str(project_root),
            "model_family": "flex",
            "model_label": label,
            "model": model_config,
            "output_path": f"{base_output}/probe_fit",
            "dataset_split_role": "probe_fit",
            "expert_labels": expert_labels,
            **capture,
            "max_tokens": int(analysis["max_probe_fit_tokens"]),
        }
        if "dataset_config_path" in source:
            train_config.update({"dataset_config_path": source["dataset_config_path"], "dataset_role": "sae_fit"})
        else:
            train_config["examples_glob"] = source["sae_fit_glob"]
        eval_config = {
            "project_root": str(project_root),
            "model_family": "flex",
            "model_label": label,
            "model": model_config,
            "output_path": f"{base_output}/eval",
            "dataset_split_role": "eval",
            "expert_labels": expert_labels,
            **capture,
        }
        if "dataset_config_path" in source:
            eval_config.update({"dataset_config_path": source["dataset_config_path"], "dataset_role": "eval"})
        else:
            eval_config["examples_glob"] = source["eval_glob"]
        analysis_config = {
            "project_root": str(project_root),
            "model_label": label,
            "train_artifacts_path": f"{base_output}/probe_fit",
            "eval_artifacts_path": f"{base_output}/eval",
            "pretrained_sae_dir": f"{base_output}/streamed_sae",
            "output_path": f"{base_output}/analysis",
            "layers": "captured",
            "expert_labels": expert_labels,
            **analysis,
        }
        sae_fit_path = generated_dir / label / "fit_streamed_sae.json"
        train_path = generated_dir / label / "capture_probe_fit.json"
        eval_path = generated_dir / label / "capture_eval.json"
        analysis_path = generated_dir / label / "analysis.json"
        _write_json(sae_fit_path, sae_fit_config)
        _write_json(train_path, train_config)
        _write_json(eval_path, eval_config)
        _write_json(analysis_path, analysis_config)
        jobs.append(
            {
                "name": label,
                "fit_sae_config": str(sae_fit_path),
                "capture_probe_config": str(train_path),
                "capture_eval_config": str(eval_path),
                "analysis_config": str(analysis_path),
            }
        )

    batch_path = generated_dir / "batch.json"
    _write_json(
        batch_path,
        {
            "project_root": str(project_root),
            "batch_name": batch_name,
            "continue_on_error": bool(matrix.get("continue_on_error", True)),
            "dataset_config_path": source.get("dataset_config_path"),
            "jobs": jobs,
        },
    )
    print(batch_path)
    if args.run:
        command = [sys.executable, str(ROOT / "scripts" / "routerinterp" / "run_routerinterp_batch.py"), "--config", str(batch_path)]
        raise SystemExit(subprocess.call(command, cwd=project_root))


if __name__ == "__main__":
    main()
