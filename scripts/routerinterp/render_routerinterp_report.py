#!/usr/bin/env python3
"""Render direct-evidence RouterInterp tables and Plotly figures from summaries."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--summary", action="append", required=True, help="RouterInterp analysis summary.json; repeat for models.")
    parser.add_argument("--output", required=True, help="Relative path under project tmp/ for report artifacts.")
    args = parser.parse_args()
    summary_paths = [_project_tmp_path(value) for value in args.summary]
    for path in summary_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing RouterInterp summary: {path}")
    print(json.dumps(render_routerinterp_report(summary_paths, _project_tmp_path(args.output)), indent=2))


if __name__ == "__main__":
    main()
