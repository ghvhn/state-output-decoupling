@echo off
setlocal enabledelayedexpansion
rem ============================================================================
rem  One-shot benchmark bootstrap for the TDA / humble-reasoner model.
rem
rem  Usage (from the repo root):
rem      run_benchmark.cmd                 :: 3-row smoke run, hardware-auto load
rem      run_benchmark.cmd 25              :: 25 rows
rem      run_benchmark.cmd 25 4bit         :: 25 rows, force low-VRAM 4-bit load
rem      run_benchmark.cmd 25 cpu          :: 25 rows, force CPU (no GPU needed)
rem      run_benchmark.cmd 5 auto small    :: open Qwen-1.5B fallback, no license
rem
rem  Load mode "auto" (default) detects your hardware and picks full / 4bit /
rem  slow-offload / cpu automatically. You can override with the 2nd argument.
rem
rem  SMALL mode (3rd arg "small", or set TDA_SMALL=1): runs the open, ungated
rem  Qwen2.5-1.5B-Instruct instead of Llama. No HF license, no 16 GB download,
rem  runs on CPU or a tiny GPU. NOTE: the cognitive cache is calibrated for the
rem  8B model and goes inert on Qwen -- you are benchmarking the reasoning
rem  scaffold on a stock model, not the cached/steered model.
rem
rem  Prereqs for the DEFAULT (Llama-3.1-8B) path -- not needed in SMALL mode:
rem    1. A CUDA GPU (or use load mode cpu / SMALL mode).
rem    2. Accept the Llama-3.1 license:
rem         https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
rem    3. An HF token: run `huggingface-cli login` once, or set HF_TOKEN=hf_xxxx
rem ============================================================================

cd /d "%~dp0"

set "N_ROWS=%~1"
if "%N_ROWS%"=="" set "N_ROWS=3"

set "LOAD_MODE=%~2"
if "%LOAD_MODE%"=="" set "LOAD_MODE=auto"

if /i "%~3"=="small" set "TDA_SMALL=1"

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

if defined TDA_SMALL (
    echo.
    echo === [3/5] SMALL mode: open Qwen2.5-1.5B-Instruct, no license/token needed ===
    echo === [4/5] Model downloads automatically on first run ^(~3 GB^) ===
    goto :run
)

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
        echo   Or run SMALL mode instead:  run_benchmark.cmd %N_ROWS% %LOAD_MODE% small
        goto :fail
    )
    echo Logged in to Hugging Face.
)

echo.
echo === [4/5] Pre-downloading model weights (~16 GB, first run only) ===
echo     The runner loads the model local-files-only, so it must be cached first.
"%PY%" -m huggingface_hub.commands.huggingface_cli download "%MODEL%" || goto :fail

:run
echo.
echo === [5/5] Running benchmark (%N_ROWS% rows, load mode: %LOAD_MODE%) ===
rem  Easter egg is left ON (default): the interactive success shell launches
rem  after the final summary is written and the model/runtime is released.
rem  Pass --boring (or --no-launch-interactive-on-success) here to suppress it
rem  for unattended runs.
set "SMALL_FLAG="
if defined TDA_SMALL set "SMALL_FLAG=--small"
"%PY%" scripts\evaluate_humble_full_suite.py ^
    --n %N_ROWS% ^
    --methods compact,humble_synthesis ^
    --run-kind bench-standard ^
    --oracle-cache-mode ignore_oracle ^
    --load-mode %LOAD_MODE% ^
    %SMALL_FLAG% ^
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
