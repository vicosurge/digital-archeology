#!/bin/bash

# Digital Archeology - Run Script
# The UI is server-rendered (HTMX + Jinja2) and served by FastAPI itself,
# so there is only one process to start: uvicorn. No Node/npm required.
#
# Note: docker compose is the primary way to run this. See the DATA note below.

set -e

cd "$(dirname "$0")"

# Load .env so OLLAMA_HOST / OLLAMA_API_KEY / DA_DATA_DIR apply here too.
if [ -f .env ]; then
    set -a; . ./.env; set +a
fi

echo "============================================"
echo "    Digital Archeology"
echo "============================================"

# --- Data location -------------------------------------------------------
# The corpus and database live in ./data (mounted at /app/data under Docker).
# Without DA_DATA_DIR the app falls back to backend/data, which on this machine
# is an EMPTY database -- the app starts fine and shows nothing at all.
export DA_DATA_DIR="${DA_DATA_DIR:-$PWD/data}"
echo "Data directory: ${DA_DATA_DIR}"

if [ ! -f "${DA_DATA_DIR}/digital_archeology.db" ]; then
    echo "  NOTE: no database there yet; a new empty one will be created."
else
    echo "  Database: $(du -h "${DA_DATA_DIR}/digital_archeology.db" | cut -f1)"
fi

# --- Ollama --------------------------------------------------------------
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
echo "Checking Ollama at ${OLLAMA_HOST}..."
if [ -n "${OLLAMA_API_KEY}" ]; then
    AUTH_ARGS=(-H "${OLLAMA_AUTH_HEADER:-Authorization}: ${OLLAMA_AUTH_SCHEME:-Bearer} ${OLLAMA_API_KEY}")
    echo "  (sending an auth header)"
else
    AUTH_ARGS=()
fi
if curl -s -f "${AUTH_ARGS[@]}" "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
    echo "  Ollama is reachable"
else
    echo "  WARNING: Ollama is not reachable. Analysis will be unavailable until it is."
    echo "           Start it with: ollama serve (or set OLLAMA_HOST / OLLAMA_API_KEY)"
fi

# Setup Python virtual environment if needed
if [ ! -d "backend/venv" ]; then
    echo ""
    echo "Setting up Python virtual environment..."
    python3 -m venv backend/venv
    source backend/venv/bin/activate
    pip install -r backend/requirements.txt
else
    source backend/venv/bin/activate
fi

# --- Reload ---------------------------------------------------------------
# --reload restarts uvicorn whenever a file changes, which kills an in-progress
# analysis run. Fine while developing, wrong for a long bulk job, so it is
# opt-in: RELOAD=1 ./run.sh
RELOAD_ARGS=()
if [ "${RELOAD:-0}" = "1" ]; then
    RELOAD_ARGS=(--reload)
    echo ""
    echo "  Auto-reload ON - do not use this for a bulk analysis run."
fi

HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")

echo ""
echo "============================================"
echo "  App:      http://localhost:8000"
echo "  API docs: http://localhost:8000/docs"
echo "  External: http://${HOST_IP}:8000"
echo "============================================"
echo ""
echo "Press Ctrl+C to stop."
echo ""

cd backend
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 "${RELOAD_ARGS[@]}"
