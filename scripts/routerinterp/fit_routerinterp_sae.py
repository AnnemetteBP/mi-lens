#!/usr/bin/env python3
"""Stream pooled router inputs into Top-K SAE fitting without saving activations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mi_lens.pipelines import run_routerinterp_streaming_sae_fit_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a streaming SAE-fit JSON config.")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(run_routerinterp_streaming_sae_fit_pipeline(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
