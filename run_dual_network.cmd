@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [Error] .venv not found. Please run run_phenomenality_shell.cmd first to set up .venv.
    pause
    exit /b 1
)

echo.
echo Launching Dual-Network Interactive Shell (CEN + DMN)...
"%PY%" "scripts\run_dual_network_shell.py" %*

pause
