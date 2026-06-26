#!/usr/bin/env bash
# Cross-platform setup script for Linux + macOS.
# Creates a venv, installs the package in editable mode with dev extras.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "==> Using Python: $($PYTHON --version)"

if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating venv at $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip"
python -m pip install --upgrade pip wheel setuptools

echo "==> Installing edge-llm-bench (editable, dev extras)"
pip install -e ".[dev]"

echo "==> Verifying install"
python -c "import edge_llm_bench; print('edge-llm-bench', edge_llm_bench.__version__)"

echo "==> Smoke test"
pytest tests/ -q --no-header || true

echo
echo "Setup complete. Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "Optional backends:"
echo "  pip install -e '.[llamacpp]'   # llama.cpp via Python bindings"
echo "  pip install -e '.[cuda]'       # NVIDIA: pynvml, auto-gptq, autoawq"
echo "  pip install -e '.[apple]'      # Apple Silicon: mlx-lm"
echo
echo "Optional: download eval datasets (one-time)"
echo "  python scripts/download_datasets.py"
