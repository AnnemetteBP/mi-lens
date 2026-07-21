#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT"
python "$ROOT/scripts/tuned_lens/run_train_tuned_lens.py" --config "$1"
