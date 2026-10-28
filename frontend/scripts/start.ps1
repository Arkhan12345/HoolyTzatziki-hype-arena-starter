# One command for the demo: install the UI, start P2's API (if present) on :8000, start the UI on :5173.
# Usage (from anywhere): frontend\scripts\start.ps1      API module override: $env:API_APP = 'app.api.main:app'
$ErrorActionPreference = 'Stop'
$Root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$ApiApp = if ($env:API_APP) { $env:API_APP } else { 'app.api.main:app' }
$Py = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $Py) { $Py = Get-Command py -ErrorAction SilentlyContinue }

Set-Location (Join-Path $Root 'frontend')
if (-not (Test-Path node_modules)) { npm install }

$Api = $null
if ($null -ne $Py -and (Test-Path (Join-Path $Root 'app\api'))) {
  Write-Host "Starting API ($ApiApp) on http://localhost:8000"
  $Api = Start-Process -FilePath $Py.Source -ArgumentList '-m', 'uvicorn', $ApiApp, '--port', '8000' -WorkingDirectory $Root -PassThru -NoNewWindow
} else {
  Write-Host 'No app\api or no Python: the UI will use mock data.'
}

try {
  Write-Host 'Starting UI on http://localhost:5173'
  npm run dev
} finally {
  if ($null -ne $Api -and -not $Api.HasExited) { Stop-Process -Id $Api.Id }
}
