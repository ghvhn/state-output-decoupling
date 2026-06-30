@echo off
setlocal enabledelayedexpansion
rem ============================================================================
rem  One-shot benchmark bootstrap for the TDA / humble-reasoner model.
rem
rem  Usage (from the repo root):
rem      run_benchmark.cmd                 :: 3-row smoke run, auto load mode
rem      run_benchmark.cmd 25              :: 25 rows
rem      run_benchmark.cmd 25 4bit         :: 25 rows, low-VRAM 4-bit load
rem
rem  Prereqs the script CANNOT do for you:
rem    1. A CUDA-capable GPU + recent NVIDIA driver.
rem    2. You must have accepted the Llama-3.1 license on Hugging Face:
rem         https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
rem    3. A Hugging Face token. Either run `huggingface-cli login` once, or set
rem         set HF_TOKEN=hf_xxxxxxxx
rem       before running this script.
rem ============================================================================

cd /d "%~dp0"

set "N_ROWS=%~1"
if "%N_ROWS%"=="" set "N_ROWS=3"

set "LOAD_MODE=%~2"
if "%LOAD_MODE%"=="" set "LOAD_MODE=auto"

set "MODEL=meta-llama/Llama-3.1-8B-Instruct"
set "PY=.venv\Scripts\python.exe"

echo.
echo === [1/5] Python virtual environment ===
if not exist "%PY%" (
    echo Creating .venv ...
    python -m venv .venv || goto :fail
) else (
    echo Reusing existing .venv
)

echo.
echo === [2/5] Installing benchmark dependencies ===
"%PY%" -m pip install --upgrade pip >nul
"%PY%" -m pip install -r requirements-bench.txt || goto :fail

echo.
echo === [3/5] Checking Hugging Face authentication ===
if defined HF_TOKEN (
    echo Using HF_TOKEN from environment.
) else (
    "%PY%" -m huggingface_hub.commands.huggingface_cli whoami >nul 2>&1
    if errorlevel 1 (
        echo.
        echo   You are not logged in to Hugging Face and HF_TOKEN is not set.
        echo   Run one of these first, then re-run this script:
        echo       %PY% -m huggingface_hub.commands.huggingface_cli login
        echo     - or -
        echo       set HF_TOKEN=hf_your_token_here
        goto :fail
    )
    echo Logged in to Hugging Face.
)

echo.
echo === [4/5] Pre-downloading model weights (~16 GB, first run only) ===
echo     The runner loads the model local-files-only, so it must be cached first.
"%PY%" -m huggingface_hub.commands.huggingface_cli download "%MODEL%" || goto :fail

echo.
echo === [5/5] Running benchmark (%N_ROWS% rows, load mode: %LOAD_MODE%) ===
rem  Easter egg is left ON (default): the interactive success shell launches
rem  after the final summary is written and the model/runtime is released.
rem  Pass --boring (or --no-launch-interactive-on-success) here to suppress it
rem  for unattended runs.
"%PY%" scripts\evaluate_humble_full_suite.py ^
    --n %N_ROWS% ^
    --methods compact,humble_synthesis ^
    --run-kind bench-standard ^
    --oracle-cache-mode ignore_oracle ^
    --load-mode %LOAD_MODE% ^
    --output invariants\out\bench_bootstrap.json || goto :fail

echo.
echo === DONE ===
echo Results written to invariants\out\bench_bootstrap.json
echo Summarize anytime with:  %PY% scripts\summarize_results.py
goto :eof

:fail
echo.
echo *** Bootstrap failed. See the message above. ***
exit /b 1
