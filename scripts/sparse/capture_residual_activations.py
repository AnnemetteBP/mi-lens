#!/usr/bin/env python3
"""Capture sharded residual activations for SAE-style analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mi_lens.pipelines import run_sparse_capture_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture reusable residual activation shards for sparse methods."
    )
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(run_sparse_capture_pipeline(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
