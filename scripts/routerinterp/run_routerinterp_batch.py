#!/usr/bin/env python3
"""Run a resumable, sequential RouterInterp batch from one JSON manifest.

Each capture or SAE-analysis stage runs in its own Python process.  This keeps
large model allocations isolated: when a stage exits, Python objects and CUDA
allocator state are released before the next model starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STAGE_SCRIPTS = {
    "fit_sae": ROOT / "scripts" / "routerinterp" / "fit_routerinterp_sae.py",
    "capture_probe": ROOT / "scripts" / "routerinterp" / "capture_routerinterp.py",
    "capture_eval": ROOT / "scripts" / "routerinterp" / "capture_routerinterp.py",
    "analysis": ROOT / "scripts" / "routerinterp" / "analyze_routerinterp.py",
}
STAGE_OUTPUT_FILE = {
    "fit_sae": "sae_fit_manifest.json",
    "capture_probe": "manifest.json",
    "capture_eval": "manifest.json",
    "analysis": "summary.json",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value).strip("_") or "job"


def _config_hash(config_path: Path) -> str:
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path)


def _stage_output(project_root: Path, stage_config: dict[str, Any]) -> Path:
    if "output_path" not in stage_config:
        raise ValueError("Every batch stage config must set `output_path` explicitly.")
    output_path = _resolve(project_root, str(stage_config["output_path"])).resolve()
    tmp_root = (project_root / "tmp").resolve()
    if output_path != tmp_root and tmp_root not in output_path.parents:
        raise ValueError("RouterInterp batch outputs must be under project_root/tmp/.")
    return output_path


def _completed(marker_path: Path, config_hash: str) -> bool:
    if not marker_path.is_file():
        return False
    try:
        marker = _load_json(marker_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    output_path = Path(str(marker.get("output_path", "")))
    required_file = str(marker.get("required_file", ""))
    return (
        marker.get("status") == "complete"
        and marker.get("config_sha256") == config_hash
        and bool(required_file)
        and (output_path / required_file).is_file()
    )


def _run_stage(
    *,
    project_root: Path,
    batch_name: str,
    job_name: str,
    stage: str,
    config_path: Path,
    resume: bool,
) -> str:
    if stage not in STAGE_SCRIPTS:
        raise ValueError(f"Unknown RouterInterp stage {stage!r}.")
    stage_config = _load_json(config_path)
    config_digest = _config_hash(config_path)
    output_path = _stage_output(project_root, stage_config)
    required_file = STAGE_OUTPUT_FILE[stage]
    state_dir = project_root / "tmp" / "routerinterp" / "batch_state" / _safe_name(batch_name) / _safe_name(job_name)
    marker_path = state_dir / f"{stage}.json"
    log_path = state_dir / f"{stage}.log"

    if resume and _completed(marker_path, config_digest):
        print(f"[resume] {job_name}: {stage} already complete")
        return "skipped"

    state_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(STAGE_SCRIPTS[stage]), "--config", str(config_path)]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = str(project_root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    print(f"[start] {job_name}: {stage}; log: {log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{_now()}] $ {' '.join(command)}\n")
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        _write_json_atomic(
            marker_path,
            {
                "status": "failed",
                "finished_at": _now(),
                "config_path": str(config_path),
                "config_sha256": config_digest,
                "output_path": str(output_path),
                "required_file": required_file,
                "log_path": str(log_path),
                "returncode": completed.returncode,
            },
        )
        raise RuntimeError(f"{job_name}: {stage} failed (exit {completed.returncode}); see {log_path}")
    if not (output_path / required_file).is_file():
        raise RuntimeError(f"{job_name}: {stage} exited successfully but did not write {output_path / required_file}")
    _write_json_atomic(
        marker_path,
        {
            "status": "complete",
            "finished_at": _now(),
            "config_path": str(config_path),
            "config_sha256": config_digest,
            "output_path": str(output_path),
            "required_file": required_file,
            "log_path": str(log_path),
        },
    )
    print(f"[done] {job_name}: {stage}", flush=True)
    return "complete"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a resumable sequential RouterInterp model batch.")
    parser.add_argument("--config", required=True, help="Path to a RouterInterp batch JSON file.")
    parser.add_argument("--no-resume", action="store_true", help="Re-run stages even if their completion marker matches.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue with later models after a failed stage.")
    args = parser.parse_args()

    batch_config_path = Path(args.config).resolve()
    batch_config = _load_json(batch_config_path)
    project_value = Path(str(batch_config.get("project_root", ROOT)))
    project_root = project_value if project_value.is_absolute() else (batch_config_path.parent.parent.parent / project_value)
    project_root = project_root.resolve()
    batch_name = str(batch_config.get("batch_name", batch_config_path.stem))
    jobs = batch_config.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Batch config must contain a non-empty `jobs` list.")

    dataset_config_value = batch_config.get("dataset_config_path")
    if dataset_config_value:
        from mi_lens.methods.router_data_prep import warm_router_dataset_cache_from_config

        dataset_config_path = _resolve(project_root, str(dataset_config_value)).resolve()
        if not dataset_config_path.is_file():
            raise FileNotFoundError(f"Missing router dataset config: {dataset_config_path}")
        print(f"[start] warming shared dataset cache from {dataset_config_path}", flush=True)
        cache_manifest = warm_router_dataset_cache_from_config(
            dataset_config_path,
            project_root=project_root,
        )
        cache_manifest_path = (
            project_root / "tmp" / "routerinterp" / "batch_state" / _safe_name(batch_name) / "dataset_cache.json"
        )
        _write_json_atomic(cache_manifest_path, cache_manifest)
        print(f"[done] shared dataset cache: {cache_manifest['cache_dir']}", flush=True)

    continue_on_error = bool(batch_config.get("continue_on_error", False) or args.continue_on_error)
    failures: list[dict[str, str]] = []
    for job_index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError(f"Job {job_index} must be a JSON object.")
        job_name = str(job.get("name", f"job_{job_index:02d}"))
        try:
            for stage in ("fit_sae", "capture_probe", "capture_eval", "analysis"):
                config_value = job.get(f"{stage}_config")
                if not config_value:
                    raise ValueError(f"{job_name} is missing `{stage}_config`.")
                stage_config_path = _resolve(project_root, str(config_value)).resolve()
                if not stage_config_path.is_file():
                    raise FileNotFoundError(f"Missing {stage} config: {stage_config_path}")
                _run_stage(
                    project_root=project_root,
                    batch_name=batch_name,
                    job_name=job_name,
                    stage=stage,
                    config_path=stage_config_path,
                    resume=not args.no_resume,
                )
        except Exception as exc:
            failures.append({"job": job_name, "error": str(exc)})
            print(f"[failed] {job_name}: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc()
            if not continue_on_error:
                break

    if failures:
        summary_path = project_root / "tmp" / "routerinterp" / "batch_state" / _safe_name(batch_name) / "failures.json"
        _write_json_atomic(summary_path, {"batch_name": batch_name, "failures": failures})
        raise SystemExit(1)
    print(f"[complete] RouterInterp batch {batch_name}", flush=True)


if __name__ == "__main__":
    main()
