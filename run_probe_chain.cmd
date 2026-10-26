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
echo ====================================================================
echo   Probe Depth ^& Layerwise Steering Chain Experiment
echo ====================================================================
echo.

"%PY%" "scripts\experiment_probe_depth_chain.py" %*

echo.
pause
