@echo off
rem ============================================================================
rem  Double-click this file to run a benchmark. No coding needed.
rem
rem  It runs the small, open Qwen model on your CPU -- no GPU, no Hugging Face
rem  account, no license. The first run downloads ~3 GB and then scores a few
rem  rows. Later you can run the full Llama model with run_benchmark.cmd.
rem ============================================================================
cd /d "%~dp0"

echo.
echo ====================================================================
echo   TDA benchmark - easy start (small model, CPU, no setup needed)
echo ====================================================================
echo.

echo Checking for Python...
where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo   Python is not installed, or it is not on your PATH.
    echo.
    echo   1. Open https://www.python.org/downloads/ in your browser
    echo   2. Download Python 3.11 or newer and run the installer
    echo   3. On the FIRST installer screen, TICK "Add python.exe to PATH"
    echo   4. Finish the install, then double-click this file again.
    echo.
    pause
    exit /b 1
)
echo Found Python.
echo.
echo This will now set things up and run a 3-row benchmark on the CPU.
echo The first run downloads about 3 GB, so it can take a while. That's normal.
echo.
pause

call "%~dp0run_benchmark.cmd" 3 cpu small

echo.
echo ====================================================================
echo   Finished. Results are in invariants\out\bench_bootstrap.json
echo   You can close this window now.
echo ====================================================================
pause
