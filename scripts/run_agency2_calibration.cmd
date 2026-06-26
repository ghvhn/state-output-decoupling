@echo off
setlocal

set "PYEXE=%TDA_PYTHON%"
if "%PYEXE%"=="" set "PYEXE=C:\Windows\System32\unsloth_env\Scripts\python.exe"

cd /d "%~dp0\.."
echo Using Python: %PYEXE%
"%PYEXE%" -u -m invariants.agency2 --calibration-only meta-llama/Llama-3.1-8B-Instruct
