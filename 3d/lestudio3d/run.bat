@echo off
cd /d "%~dp0"
if not exist .venv (
  echo Creating local environment...
  python -m venv .venv
  call .venv\Scripts\activate.bat
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
python app.py
pause
