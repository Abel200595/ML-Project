#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
VENV_STREAMLIT="$ROOT/.venv/bin/streamlit"
if [[ ! -x "$VENV_STREAMLIT" ]]; then
  echo "Missing $VENV_STREAMLIT"
  echo "Create the venv and install deps from the repo root:"
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/pip install -r requirements.txt"
  exit 1
fi
exec "$VENV_STREAMLIT" run app.py "$@"
