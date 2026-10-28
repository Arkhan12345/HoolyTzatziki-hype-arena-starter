# Start the demo API for the frontend on http://localhost:8000
Set-Location (Join-Path $PSScriptRoot "..")
$port = if ($env:PORT) { $env:PORT } else { "8000" }
$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
& $py -m uvicorn app.api.main:app --host 127.0.0.1 --port $port --reload
