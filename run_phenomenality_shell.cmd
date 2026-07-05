@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

set "PY=.venv\Scripts\python.exe"

echo.
echo === [1/3] Python virtual environment ===
if not exist "%PY%" (
    echo Creating .venv ...
    python -m venv .venv || goto :fail
) else (
    echo Reusing existing .venv
)

echo.
echo === [2/3] Installing dependencies ===
"%PY%" -m pip install --upgrade pip >nul
"%PY%" -m pip install -r requirements-bench.txt || goto :fail

echo.
echo === [3/3] Checking Hugging Face authentication ===
if defined HF_TOKEN (
    echo Using HF_TOKEN from environment.
    goto :auth_ok
)
"%PY%" -c "import sys; from huggingface_hub import get_token; sys.exit(0 if get_token() else 1)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo   No Hugging Face token found. The shell may fall back to smaller open models
    echo   or fail if you explicitly request a gated model without logging in.
    echo   To log in once: %PY% -c "from huggingface_hub import login; login()"
    echo.
) else (
    echo Hugging Face token detected.
)

:auth_ok
echo.
echo === Launching Phenomenality Shell ===
rem  Optional first argument: a model id/path to run the egg shell on, e.g.
rem      run_phenomenality_shell.cmd Qwen/Qwen2.5-1.5B-Instruct
"%PY%" "scripts\interactive_phenomenality.py" %*

echo.
echo [Phenomenality shell exited. Press any key to close.]
pause >nul
goto :eof

:fail
echo.
echo *** Setup failed. See the message above. ***
pause >nul
exit /b 1
