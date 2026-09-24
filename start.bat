@echo off
rem SIH Retail Intelligence - one-command localhost launcher (Windows)
rem Copy this folder to any machine and double-click start.bat (or run it).
cd /d "%~dp0"

set "PY=python"
where python >nul 2>nul
if errorlevel 1 (
  where py >nul 2>nul
  if errorlevel 1 (
    echo Python not found on PATH. Install Python 3.9+ from https://www.python.org/downloads/
    echo IMPORTANT: tick "Add python.exe to PATH" during install, then restart this file.
    pause
    exit /b 1
  )
  set "PY=py -3"
)

if not exist .venv (
  echo [1/3] Creating virtual environment...
  %PY% -m venv .venv
)

call .venv\Scripts\activate.bat

echo [2/3] Installing dependencies ^(first run only, may take a few minutes^)...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo [3/3] Starting dashboard on http://localhost:8000 ...
start "" python run_web.py --host 127.0.0.1 --port 8000

timeout /t 4 /nobreak >nul
start http://localhost:8000/

echo.
echo Dashboard starting at http://localhost:8000
echo Keep this window open. Press Ctrl+C twice to stop the server.
pause