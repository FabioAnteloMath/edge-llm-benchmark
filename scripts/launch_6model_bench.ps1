# Launch the 6-model benchmark in background.
$ErrorActionPreference = "Stop"

$python = "C:\WorkSpace\Pessoal\edge-llm-benchmark\.venv\Scripts\python.exe"
$workdir = "C:\WorkSpace\Pessoal\edge-llm-benchmark"
$outdir = Join-Path $workdir "results\smoke-2026-06-26-6models"
$stdoutLog = Join-Path $outdir "run-stdout.txt"
$stderrLog = Join-Path $outdir "run-stderr.txt"

New-Item -ItemType Directory -Force -Path $outdir | Out-Null
"" | Set-Content $stdoutLog -Encoding UTF8
"" | Set-Content $stderrLog -Encoding UTF8

$proc = Start-Process -FilePath $python `
    -ArgumentList @(
        "-m", "edge_llm_bench.runner",
        "--profile", "configs/profiles/windows-dev-laptop.yaml",
        "--config", "configs/matrix-smoke.yaml",
        "--max-examples", "2",
        "--output-dir", $outdir
    ) `
    -WorkingDirectory $workdir `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Benchmark PID: $($proc.Id)"
Write-Host "Working dir:   $workdir"
Write-Host "Output dir:    $outdir"
Write-Host "Stdout log:    $stdoutLog"
Write-Host "Stderr log:    $stderrLog"
Write-Host "Started:       $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
