# Windows equivalent of start-lab.sh: starts the worker (8766) and UI (3000).
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$required = 'visual-circuit.npz','full-spiking.npz','full-brain.json','full-positions.bin','full-graded.npz','full-retina.json' |
  ForEach-Object { Join-Path 'data\malecns' $_ }
if (-not (Test-Path $py) -or -not (Test-Path 'ui\node_modules') -or ($required | Where-Object { -not (Test-Path $_) })) {
  Write-Error 'Dependencies or circuit data are missing. Follow README.md setup first.'
}
foreach ($port in 8766, 3000) {
  if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
    Write-Error "Port $port is occupied. Stop the existing lab before starting another."
  }
}
New-Item -ItemType Directory -Force .lab-logs | Out-Null
$worker = Start-Process $py -ArgumentList '-m','uvicorn','lab.server:app','--host','127.0.0.1','--port','8766','--no-access-log' `
  -RedirectStandardOutput .lab-logs\worker.log -RedirectStandardError .lab-logs\worker.err.log -NoNewWindow -PassThru
$ui = Start-Process npm.cmd -ArgumentList 'run','dev','--','--host','127.0.0.1','--port','3000','--strictPort' -WorkingDirectory ui `
  -RedirectStandardOutput .lab-logs\ui.log -RedirectStandardError .lab-logs\ui.err.log -NoNewWindow -PassThru
Write-Output 'flyByWire: http://localhost:3000/'
Write-Output 'Logs: .lab-logs\worker.log and .lab-logs\ui.log. Ctrl-C to stop.'
try {
  while (-not $worker.HasExited -and -not $ui.HasExited) { Start-Sleep -Seconds 1 }
  Write-Warning 'A service stopped. See .lab-logs\ for details.'
} finally {
  foreach ($p in $worker, $ui) { if (-not $p.HasExited) { taskkill /PID $p.Id /T /F | Out-Null } }
}
