@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem ???????PIG_PYTHON ???? > ?? conda ?? > PATH ?? python
set "PY=%PIG_PYTHON%"
if not defined PY set "PY=C:\Users\Programs\Miniconda3\envs\data_science\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================================
echo   Pig Phenotype Agent - one click launcher
echo   (close this window or press Ctrl+C to stop the service)
echo ============================================================
echo.

"%PY%" "scripts\start_web.py" %*

if errorlevel 1 (
  echo.
  echo [Startup failed] See the messages above.
  pause
)

endlocal
