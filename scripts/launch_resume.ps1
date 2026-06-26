# Launch the benchmark runner in the background with --resume.
$ErrorActionPreference = "Stop"

$python = "C:\WorkSpace\Pessoal\edge-llm-benchmark\.venv\Scripts\python.exe"
$workdir = "C:\WorkSpace\Pessoal\edge-llm-benchmark"
$outdir = "C:\WorkSpace\Pessoal\edge-llm-benchmark\results\smoke-2026-06-25"
$stdoutLog = Join-Path $outdir "run-resume.txt"
$stderrLog = Join-Path $outdir "run-resume.err.txt"

# Truncate prior log
"" | Set-Content $stdoutLog -Encoding UTF8
"" | Set-Content $stderrLog -Encoding UTF8

$psi = Start-Process -FilePath $python `
    -ArgumentList @(
        "-m", "edge_llm_bench.runner",
        "--profile", "configs/profiles/windows-dev-laptop.yaml",
        "--config", "configs/matrix-smoke.yaml",
        "--max-examples", "3",
        "--output-dir", "results/smoke-2026-06-25",
        "--resume"
    ) `
    -WorkingDirectory $workdir `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru

Write-Host "PID: $($psi.Id)"
Write-Host "Logs: $stdoutLog"
Write-Host "Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
