@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Run scripts\setup_and_run.bat first.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
set "PYTHONPATH=%CD%\code"
python -m predictive_maintenance.cli analyze --sos data\current\SosFluidSample.xlsx --telemetry data\current\TelematicDataSample.xlsx --output outputs\current
pause
endlocal
