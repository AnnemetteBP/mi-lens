#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT"
python "$ROOT/scripts/analysis/run_tokenization_audit.py" --config "$1"
