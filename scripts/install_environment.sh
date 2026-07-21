#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
FLEX_TRANSFORMERS_DIR=""
TORCH_INDEX_URL=""

usage() {
    cat <<'EOF'
Usage:
  conda env create -f environment.yml
  conda activate mi-lens
  scripts/install_environment.sh [options]

Options:
  --flex-transformers PATH  Install a checked-out custom Transformers fork.
  --torch-index URL         Install PyTorch from this index before mi-lens.
  -h, --help                Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --flex-transformers)
            FLEX_TRANSFORMERS_DIR="${2:?--flex-transformers requires a path}"
            shift 2
            ;;
        --torch-index)
            TORCH_INDEX_URL="${2:?--torch-index requires a URL}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

"$PYTHON" - <<'PY'
import sys

if not ((3, 10) <= sys.version_info < (3, 14)):
    raise SystemExit("mi-lens requires Python >=3.10,<3.14; use environment.yml (Python 3.12).")
PY

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/tmp/pip-cache}"
mkdir -p "$PIP_CACHE_DIR"

"$PYTHON" -m pip install --upgrade pip setuptools wheel
if [[ -n "$TORCH_INDEX_URL" ]]; then
    "$PYTHON" -m pip install --index-url "$TORCH_INDEX_URL" "torch>=2.2"
fi
"$PYTHON" -m pip install -e "$ROOT[dev,notebooks,slow_tokenizers]"

if [[ -n "$FLEX_TRANSFORMERS_DIR" ]]; then
    if [[ ! -f "$FLEX_TRANSFORMERS_DIR/pyproject.toml" && ! -f "$FLEX_TRANSFORMERS_DIR/setup.py" ]]; then
        echo "Not a Transformers source checkout: $FLEX_TRANSFORMERS_DIR" >&2
        exit 2
    fi
    "$PYTHON" -m pip install -e "$FLEX_TRANSFORMERS_DIR"
fi

"$PYTHON" - <<'PY'
import datasets
import torch
import torchdata
import transformers

print("Python:", __import__("sys").version.split()[0])
print("Torch:", torch.__version__)
print("TorchData:", torchdata.__version__)
print("Datasets:", datasets.__version__)
print("Transformers:", transformers.__version__)
print("Transformers path:", transformers.__file__)
try:
    from torchdata import dataloader2
except ImportError as exc:
    raise SystemExit("torchdata.dataloader2 is unavailable; install torchdata<0.8.") from exc
PY
