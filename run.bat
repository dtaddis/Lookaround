@echo off
rem Lookaround: starts the web app (and ComfyUI, if it isn't running) and opens it in your browser.
rem Close this window to stop both.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Lookaround isn't installed yet. Run install.bat first.
    pause
    exit /b 1
)
start "" http://127.0.0.1:8000/
".venv\Scripts\python.exe" -m uvicorn server:app --host 127.0.0.1 --port 8000
if errorlevel 1 pause
