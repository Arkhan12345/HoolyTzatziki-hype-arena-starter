#!/usr/bin/env sh
# One command for the demo: install the UI, start P2's API (if present) on :8000, start the UI on :5173.
# Usage (from anywhere): frontend/scripts/start.sh      API module override: API_APP=app.api.main:app
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
API_APP="${API_APP:-app.api.main:app}"
PY="$(command -v python3 || command -v python || true)"

cd "$ROOT/frontend"
[ -d node_modules ] || npm install

API_PID=""
if [ -n "$PY" ] && [ -d "$ROOT/app/api" ]; then
  echo "Starting API ($API_APP) on http://localhost:8000"
  (cd "$ROOT" && "$PY" -m uvicorn "$API_APP" --port 8000) &
  API_PID=$!
  trap '[ -n "$API_PID" ] && kill "$API_PID" 2>/dev/null' EXIT INT TERM
else
  echo "No app/api or no Python: the UI will use mock data."
fi

echo "Starting UI on http://localhost:5173"
npm run dev
