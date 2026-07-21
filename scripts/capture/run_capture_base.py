#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "tmp" / "hf_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "transformers")

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lenses" / "jacobian_lens"))
sys.path.insert(0, str(ROOT / "lenses" / "tuned_logit_lens"))

from mi_lens.pipelines import run_capture_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture base model tensors only.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config["lens_names"] = []
    config["save_prompt_artifacts"] = True
    result = run_capture_pipeline(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
