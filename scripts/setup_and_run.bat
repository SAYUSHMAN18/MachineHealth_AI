@echo off
setlocal
cd /d "%~dp0.."

echo [1/3] Creating Python virtual environment...
if not exist ".venv\Scripts\python.exe" python -m venv .venv

echo [2/3] Installing dependencies...
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo [3/3] Running automated checks and starting dashboard...
set "PYTHONPATH=%CD%\code"
python -m pytest
if errorlevel 1 (
  echo Tests failed. Review the output above.
  pause
  exit /b 1
)
python -m streamlit run code\app.py
endlocal
