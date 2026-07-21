#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mi_lens.pipelines import run_tokenization_audit_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tokenization and fragmentation audit across languages."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_tokenization_audit_pipeline(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
