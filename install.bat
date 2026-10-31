@echo off
rem Lookaround install: Python venvs, AnySplat, ComfyUI and the H3 models.
rem Safe to run again: it updates what's there and fetches only what's missing.
rem Options (passed to the ComfyUI step): --comfy-dir DIR  --comfy-python PATH  --yes
setlocal EnableExtensions
cd /d "%~dp0"

rem Files in use can't be updated, so Lookaround (and a ComfyUI it started) must be closed.
powershell -NoProfile -Command "if (Get-Process python* -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '%~dp0*' }) { exit 1 }" || (
    echo Lookaround is running. Close its window, then run install.bat again.
    goto :failed
)
where git >nul 2>nul || (
    echo Lookaround needs Git: https://git-scm.com/download/win
    goto :failed
)
where uv >nul 2>nul || (
    echo Installing uv, which sets up Lookaround's Python venvs...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" || goto :failed
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

echo.
echo === Lookaround venv (Python 3.11: the web app, AnySplat and subject masks)
if not exist ".venv\Scripts\python.exe" uv venv --python 3.11 .venv || goto :failed
set "PY=.venv\Scripts\python.exe"
uv pip install --python %PY% torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 || goto :failed
uv pip install --python %PY% xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124 || goto :failed
uv pip install --python %PY% torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html || goto :failed
uv pip install --python %PY% -r requirements.txt || goto :failed

echo.
echo === Refine venv (Python 3.10: gsplat's prebuilt CUDA kernels)
if not exist ".venv-refine\Scripts\python.exe" uv venv --python 3.10 .venv-refine || goto :failed
set "RPY=.venv-refine\Scripts\python.exe"
uv pip install --python %RPY% torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124 || goto :failed
uv pip install --python %RPY% "https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/gsplat-1.5.3%%2Bpt24cu124-cp310-cp310-win_amd64.whl" "numpy<2" plyfile pillow packaging setuptools || goto :failed

echo.
echo === AnySplat (its weights download the first time you build a splat)
if exist "vendor\AnySplat\.git" (
    git -C vendor\AnySplat pull --ff-only || goto :failed
) else (
    git clone https://github.com/InternRobotics/AnySplat vendor\AnySplat || goto :failed
)

echo.
echo === ComfyUI and the H3 models
%PY% -m lookaround.install %* || goto :failed

echo.
echo Lookaround is installed. Start it with run.bat.
pause
exit /b 0

:failed
echo.
echo The install stopped with an error (see above). Fix it and run install.bat again.
pause
exit /b 1
