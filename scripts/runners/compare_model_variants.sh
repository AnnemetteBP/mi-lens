#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT"
python "$ROOT/scripts/analysis/run_compare_model_variants.py" --config "$1"
