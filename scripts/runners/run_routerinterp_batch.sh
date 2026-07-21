#!/usr/bin/env bash
# Run RouterInterp stages sequentially. Each stage has its own Python process.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:?usage: run_routerinterp_batch.sh CONFIG.json [additional batch arguments]}"
shift

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python || command -v python3)"
fi
exec "$PYTHON_BIN" "$ROOT/scripts/routerinterp/run_routerinterp_batch.py" --config "$CONFIG" "$@"
