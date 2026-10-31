@echo off
rem Lookaround launcher: starts the web app and opens it in your browser.
rem Needs ComfyUI running (for the H3 orbit) and the .venv set up as in README.md.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Lookaround is not set up yet. See README.md.
    pause
    exit /b 1
)
start "" http://127.0.0.1:8000/
".venv\Scripts\python.exe" -m uvicorn server:app --host 127.0.0.1 --port 8000
if errorlevel 1 pause
