#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$PROJECT_DIR"

if [[ -x "portable_Venv/bin/python" ]]; then
    PYTHON="portable_Venv/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
elif [[ -x "venv/bin/python" ]]; then
    PYTHON="venv/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

exec "$PYTHON" -m gui
