#!/usr/bin/env python3
"""Render direct-evidence RouterInterp tables and Plotly figures from summaries."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mi_lens.plotting.routerinterp_report import render_routerinterp_report


def _project_tmp_path(value: str) -> Path:
    path = Path(value)
    destination = path if path.is_absolute() else ROOT / path
    destination = destination.resolve()
    tmp_root = (ROOT / "tmp").resolve()
    if destination != tmp_root and tmp_root not in destination.parents:
        raise ValueError("RouterInterp reports must be written under project_root/tmp/.")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", help="RouterInterp analysis summary.json; repeat for models.")
    parser.add_argument(
        "--summary-glob",
        action="append",
        help="Glob for RouterInterp summaries relative to the project root; repeat if needed.",
    )
    parser.add_argument("--output", required=True, help="Relative path under project tmp/ for report artifacts.")
    args = parser.parse_args()
    if not args.summary and not args.summary_glob:
        parser.error("provide at least one --summary or --summary-glob")

    summary_values = list(args.summary or [])
    for pattern in args.summary_glob or []:
        pattern_path = Path(pattern)
        resolved_pattern = str(pattern_path if pattern_path.is_absolute() else ROOT / pattern_path)
        matches = sorted(glob.glob(resolved_pattern))
        if not matches:
            parser.error(f"--summary-glob matched no files: {pattern}")
        summary_values.extend(matches)

    summary_paths = []
    for value in summary_values:
        path = _project_tmp_path(value)
        if path not in summary_paths:
            summary_paths.append(path)
    for path in summary_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing RouterInterp summary: {path}")
    print(json.dumps(render_routerinterp_report(summary_paths, _project_tmp_path(args.output)), indent=2))


if __name__ == "__main__":
    main()
