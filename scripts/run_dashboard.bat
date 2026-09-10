@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Run scripts\setup_and_run.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -c "import sys" >nul 2>nul
if errorlevel 1 (
  echo The existing .venv is stale or broken. Run scripts\setup_and_run.bat to recreate it.
  pause
  exit /b 1
)
set "PYTHONPATH=%CD%\code"
set "PYTHONDONTWRITEBYTECODE=1"
".venv\Scripts\python.exe" -m streamlit run code\app.py
endlocal
