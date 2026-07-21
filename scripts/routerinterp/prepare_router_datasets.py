#!/usr/bin/env python3
"""Download fixed source-order router datasets into the project data directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mi_lens.methods.router_data_prep import prepare_router_datasets_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "routerinterp" / "datasets_pilot.json",
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--role",
        choices=("all", "sae_fit", "eval"),
        default="all",
        help="Export both data roles or only one of them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write accessible sources and record unavailable sources in the registry.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roles = None if args.role == "all" else {args.role}
    manifests = prepare_router_datasets_from_config(
        args.config,
        project_root=args.project_root,
        roles=roles,
        continue_on_error=args.continue_on_error,
    )
    for manifest in manifests:
        print(f"{manifest.records_written:>4} rows  {manifest.output_path}")


if __name__ == "__main__":
    main()
