@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem Order: PIG_PYTHON -> conda env python -> whatever python is on PATH
set "PY=%PIG_PYTHON%"
if not defined PY set "PY=C:\Users\Programs\Miniconda3\envs\data_science\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================================
echo   Pig Phenotype Analysis Assistant
echo   The browser opens automatically when the service is ready.
echo   Keep this window open; close it (or press Ctrl+C) to stop.
echo ============================================================
echo.

"%PY%" "scripts\start_web.py" %*

if errorlevel 1 (
  echo.
  echo [Startup failed] See the messages above.
  pause
)

endlocal
