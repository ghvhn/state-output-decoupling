@echo off
setlocal

set "PYEXE=%TDA_PYTHON%"
if "%PYEXE%"=="" set "PYEXE=C:\Windows\System32\unsloth_env\Scripts\python.exe"

cd /d "%~dp0\.."
echo Using Python: %PYEXE%
"%PYEXE%" -u -m invariants.benchmark_goldilocks
