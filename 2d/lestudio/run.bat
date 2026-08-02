@echo off
REM ============================================================================
REM  leStudio launcher -- creates .venv, installs deps (leos-core from PyPI,
REM  Flask/Pillow/NumPy, and the app itself), then starts the editor.
REM  First run does the setup; later runs skip straight to launch.
REM ============================================================================
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [leStudio] Python was not found on PATH. Install Python 3.9+ and retry.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [leStudio] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [leStudio] Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

REM Install/refresh dependencies only when the app isn't importable yet
".venv\Scripts\python.exe" -c "import lestudio, lecore" >nul 2>nul
if errorlevel 1 (
    echo [leStudio] Installing dependencies ^(leos-core, Flask, Pillow, NumPy^)...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -e .
    if errorlevel 1 (
        echo [leStudio] Dependency install failed -- see the log above.
        pause
        exit /b 1
    )
)

REM Optional accelerators: `run.bat accel` installs the safe CPU fast paths
REM (numba JIT, Zig native kernels ~2-5x, FFTW). They are NOT installed by
REM default -- the Zig toolchain wheel alone is ~45 MB, and the app runs fine
REM without any of them (it falls back to NumPy automatically).
if /I "%~1"=="accel" (
    echo [leStudio] Installing optional CPU accelerators ^(numba, ziglang, pyfftw^)...
    ".venv\Scripts\python.exe" -m pip install -e .[accel]
    echo [leStudio] Done. GPU is separate: pip install cupy-cuda12x ^(match your CUDA^).
)

echo [leStudio] Starting -- open http://127.0.0.1:5050 in your browser.
echo [leStudio] Tip: `run.bat accel` installs optional fast paths ^(2-5x on heavy nodes^).
start "" http://127.0.0.1:5050
".venv\Scripts\python.exe" -m lestudio

endlocal
