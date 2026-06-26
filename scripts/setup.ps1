#!/usr/bin/env pwsh
# Windows setup script (PowerShell).
# Creates a venv, installs the package in editable mode with dev extras.
[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "==> Using Python: $(& $Python --version)"

if (-not (Test-Path $VenvDir)) {
    Write-Host "==> Creating venv at $VenvDir"
    & $Python -m venv $VenvDir
}

$activate = Join-Path $VenvDir "Scripts\Activate.ps1"
Write-Host "==> Activating venv: $activate"
& $activate

Write-Host "==> Upgrading pip"
python -m pip install --upgrade pip wheel setuptools | Out-Null

Write-Host "==> Installing edge-llm-bench (editable, dev extras)"
pip install -e ".[dev]"

Write-Host "==> Verifying install"
python -c "import edge_llm_bench; print('edge-llm-bench', edge_llm_bench.__version__)"

Write-Host "==> Smoke test"
pytest tests/ -q --no-header
if ($LASTEXITCODE -ne 0) { Write-Host "Some tests failed — see above." }

Write-Host ""
Write-Host "Setup complete. Activate with:"
Write-Host "  .\$VenvDir\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Optional backends:"
Write-Host "  pip install -e '.[llamacpp]'   # llama.cpp via Python bindings"
Write-Host "  pip install -e '.[cuda]'       # NVIDIA: pynvml, auto-gptq, autoawq"
Write-Host "  pip install -e '.[apple]'      # Apple Silicon: mlx-lm (Mac only)"
Write-Host ""
Write-Host "Optional: download eval datasets (one-time)"
Write-Host "  python scripts\download_datasets.py"
