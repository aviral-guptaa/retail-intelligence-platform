#!/usr/bin/env bash
# SIH Retail Intelligence — one-command localhost launcher (macOS / Linux)
#
# Copies/moves this folder to any machine, then:
#     ./start.sh
# It creates its own .venv (if needed), installs the Python deps, boots the
# dashboard on http://localhost:8000 and opens your browser.
set -e
cd "$(dirname "$0")"

PY=python3
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python 3 not found. Install Python 3.9+ and retry."
  exit 1
fi

if [ ! -d .venv ]; then
  echo "[1/3] Creating virtual environment..."
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[2/3] Installing dependencies (first run only)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "[3/3] Starting dashboard on http://localhost:8000 ..."
python run_web.py --host 127.0.0.1 --port 8000 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT INT TERM

# Wait for the server to answer, then open the browser.
for _ in $(seq 1 40); do
  if curl -sf http://127.0.0.1:8000/ >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

if command -v open >/dev/null 2>&1; then      # macOS
  open http://localhost:8000/
elif command -v xdg-open >/dev/null 2>&1; then # Linux
  xdg-open http://localhost:8000/
fi

wait "$SERVER_PID"