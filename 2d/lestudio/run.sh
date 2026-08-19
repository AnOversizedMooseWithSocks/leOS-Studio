#!/usr/bin/env bash
# ============================================================================
#  leStudio launcher (macOS / Linux) -- creates .venv, installs deps
#  (leos-core from PyPI, Flask/Pillow/NumPy, and the app itself), then starts
#  the editor. First run does the setup; later runs skip straight to launch.
#
#  Usage:   ./run.sh           start the editor
#           ./run.sh accel     also install optional CPU fast paths first
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY="python3"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "[leStudio] python3 was not found on PATH. Install Python 3.9+ (e.g. from python.org or Homebrew: brew install python) and retry."
    exit 1
fi

VPY=".venv/bin/python"
if [ ! -x "$VPY" ]; then
    echo "[leStudio] Creating virtual environment..."
    "$PY" -m venv .venv || { echo "[leStudio] Failed to create the virtual environment."; exit 1; }
fi

# Install/refresh dependencies only when the app isn't importable yet
if ! "$VPY" -c "import lestudio, lecore" >/dev/null 2>&1; then
    echo "[leStudio] Installing dependencies (leos-core, Flask, Pillow, NumPy)..."
    "$VPY" -m pip install --upgrade pip
    "$VPY" -m pip install -e . || { echo "[leStudio] Dependency install failed -- see the log above."; exit 1; }
fi

# Optional accelerators: `./run.sh accel` installs the safe CPU fast paths
# (numba JIT, Zig native kernels ~2-5x, FFTW). They are NOT installed by
# default -- the Zig toolchain wheel alone is ~45 MB, and the app runs fine
# without any of them (it falls back to NumPy automatically).
if [ "${1:-}" = "accel" ]; then
    echo "[leStudio] Installing optional CPU accelerators (numba, ziglang, pyfftw)..."
    "$VPY" -m pip install -e ".[accel]"
    echo "[leStudio] Done. GPU is separate and CUDA-only (not Apple Silicon): pip install cupy-cuda12x on a CUDA machine."
fi

# Honour the same environment the app reads, or the launcher announces (and
# opens) a URL the server is not listening on.
HOST="${LESTUDIO_HOST:-127.0.0.1}"
PORT="${LESTUDIO_PORT:-5050}"
URL="http://${HOST}:${PORT}"
echo "[leStudio] Starting -- open ${URL} in your browser."
echo "[leStudio] Tip: ./run.sh accel installs optional fast paths (2-5x on heavy nodes)."
case "$(uname -s)" in                              # `open` exists on Linux too
    Darwin) open "$URL" >/dev/null 2>&1 || true ;;   # macOS
    *) command -v xdg-open >/dev/null 2>&1 && \
       xdg-open "$URL" >/dev/null 2>&1 || true ;;    # Linux
esac
exec "$VPY" -m lestudio
