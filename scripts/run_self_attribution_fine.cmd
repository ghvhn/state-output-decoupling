@echo off
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
  echo Missing local Python environment at .venv\Scripts\python.exe
  exit /b 1
)
if not exist "invariants\out" mkdir "invariants\out"
".venv\Scripts\python.exe" -u -m invariants.self_attribution_fine > "invariants\out\self_attribution_fine.log" 2>&1
exit /b %ERRORLEVEL%
