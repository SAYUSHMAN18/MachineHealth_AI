@echo off
setlocal
cd /d "%~dp0.."

set "BOOTSTRAP="
set "USE_UV="
where py >nul 2>nul
if not errorlevel 1 set "BOOTSTRAP=py -3"
if not defined BOOTSTRAP (
  where python >nul 2>nul
  if not errorlevel 1 set "BOOTSTRAP=python"
)
if not defined BOOTSTRAP (
  where uv >nul 2>nul
  if not errorlevel 1 set "USE_UV=1"
)

echo [1/3] Checking Python virtual environment...
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import sys" >nul 2>nul
  if errorlevel 1 (
    echo Existing .venv is stale or was copied from another machine. Recreating it...
    rmdir /s /q ".venv"
  )
)
if not exist ".venv\Scripts\python.exe" (
  if not defined BOOTSTRAP if not defined USE_UV (
    echo Python 3.10 or newer was not found. Install Python, then run this script again.
    pause
    exit /b 1
  )
  if defined BOOTSTRAP %BOOTSTRAP% -m venv .venv
  if defined USE_UV uv venv .venv --python 3.12
  if errorlevel 1 (
    echo Failed to create .venv.
    pause
    exit /b 1
  )
)

echo [2/3] Installing dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed.
  pause
  exit /b 1
)

echo [3/3] Running automated checks and starting dashboard...
set "PYTHONPATH=%CD%\code"
set "PYTHONDONTWRITEBYTECODE=1"
".venv\Scripts\python.exe" -m pytest
if errorlevel 1 (
  echo Tests failed. Review the output above.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run code\app.py
endlocal
