# Pull 3 small models in parallel background processes.
# Each model is small (<2 GB) but downloads serially in Ollama, so doing
# them as separate concurrent `ollama pull` invocations is faster.
$ErrorActionPreference = "Stop"

$ollama = "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe"
$logDir = "C:\WorkSpace\Pessoal\edge-llm-benchmark\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$models = @("llama3.2:1b", "gemma2:2b", "qwen2.5:3b")
$pids = @()

foreach ($m in $models) {
    $logFile = Join-Path $logDir "pull-$($m.Replace(':','-')).log"
    "" | Set-Content $logFile -Encoding UTF8
    Write-Host "Launching pull: $m (log: $logFile)"
    $proc = Start-Process -FilePath $ollama `
        -ArgumentList @("pull", $m) `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError "$logFile.err" `
        -WindowStyle Hidden `
        -PassThru
    $pids += [PSCustomObject]@{ Model = $m; Pid = $proc.Id }
}

Write-Host ""
Write-Host "PIDs:"
$pids | Format-Table -AutoSize
Write-Host "Logs in: $logDir"
