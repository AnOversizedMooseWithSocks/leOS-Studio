#!/usr/bin/env bash
# Poly Studio launcher (macOS / Linux). Creates a local venv on first run and installs the engine
# (leos-core) plus Flask/Pillow from requirements.txt, then starts the app on http://127.0.0.1:5000/.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Creating local environment (first run installs the leCore engine from pypi)…"
  python3 -m venv .venv
  . .venv/bin/activate
  pip install --upgrade pip >/dev/null
  pip install -r requirements.txt
else
  . .venv/bin/activate
fi
exec python app.py
