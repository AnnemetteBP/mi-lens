#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT"
python "$ROOT/scripts/jlens/run_fit_jlens.py" --config "$1"
