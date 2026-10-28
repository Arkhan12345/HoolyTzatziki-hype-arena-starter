#!/usr/bin/env sh
# One command for the demo: install the UI, start P2's API (if present) on :8000, start the UI on :5173.
# Usage (from anywhere): frontend/scripts/start.sh      API module override: API_APP=app.api.main:app
# Python: the repo's .venv if it exists, else python3, else python (it needs requirements.txt installed).
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
API_APP="${API_APP:-app.api.main:app}"
PORT="${PORT:-8000}"
if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"
elif [ -x "$ROOT/.venv/Scripts/python.exe" ]; then PY="$ROOT/.venv/Scripts/python.exe"
else PY="$(command -v python3 || command -v python || true)"; fi

cd "$ROOT/frontend"
[ -d node_modules ] || npm ci

API_PID=""
if [ -n "$PY" ] && [ -d "$ROOT/app/api" ]; then
  echo "Starting API ($API_APP) on http://127.0.0.1:$PORT with $PY"
  (cd "$ROOT" && "$PY" -m uvicorn "$API_APP" --host 127.0.0.1 --port "$PORT") &
  API_PID=$!
  trap '[ -n "$API_PID" ] && kill "$API_PID" 2>/dev/null' EXIT INT TERM
else
  echo 'No app/api or no Python: the UI shows the "API offline" state and can load labelled demo data.'
fi

echo "Starting UI on http://localhost:5173"
npm run dev
