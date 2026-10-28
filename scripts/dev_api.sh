#!/bin/sh
# Start the demo API for the frontend on http://localhost:8000
set -eu
cd "$(dirname "$0")/.."
exec python -m uvicorn app.api.main:app --host 127.0.0.1 --port "${PORT:-8000}" --reload
