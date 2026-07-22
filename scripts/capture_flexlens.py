#!/usr/bin/env python3
"""Capture compact FlexLens rows from a JSON configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mi_lens.pipelines.run_flex_lens import run_flex_lens_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(run_flex_lens_pipeline(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

