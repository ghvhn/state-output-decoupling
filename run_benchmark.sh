#!/usr/bin/env bash
# ============================================================================
#  One-shot benchmark bootstrap for the TDA / humble-reasoner model.
#
#  Usage (from the repo root):
#      ./run_benchmark.sh                 :: 3-row smoke run, hardware-auto load
#      ./run_benchmark.sh 25              :: 25 rows
#      ./run_benchmark.sh 25 4bit         :: 25 rows, force low-VRAM 4-bit load
#      ./run_benchmark.sh 25 cpu          :: 25 rows, force CPU (no GPU needed)
#      ./run_benchmark.sh 5 auto small    :: open Qwen-1.5B fallback, no license
# ============================================================================

cd "$(dirname "$0")"

N_ROWS="${1:-3}"
LOAD_MODE="${2:-auto}"
if [ "$3" = "small" ]; then
    export TDA_SMALL=1
fi

MODEL="meta-llama/Llama-3.1-8B-Instruct"
PY=".venv/bin/python"

echo ""
echo "=== [1/5] Python virtual environment ==="
if [ ! -f "$PY" ]; then
    echo "Creating .venv ..."
    python3 -m venv .venv || exit 1
else
    echo "Reusing existing .venv"
fi

echo ""
echo "=== [2/5] Installing benchmark dependencies ==="
"$PY" -m pip install --upgrade pip >/dev/null
"$PY" -m pip install -r requirements-bench.txt || exit 1

if [ -n "$TDA_SMALL" ]; then
    echo ""
    echo "=== [3/5] SMALL mode: open Qwen2.5-1.5B-Instruct, no license/token needed ==="
    echo "=== [4/5] Model downloads automatically on first run (~3 GB) ==="
else
    echo ""
    echo "=== [3/5] Checking Hugging Face authentication ==="
    if [ -n "$HF_TOKEN" ]; then
        echo "Using HF_TOKEN from environment."
    else
        "$PY" -c "import sys; from huggingface_hub import get_token; sys.exit(0 if get_token() else 1)" >/dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo ""
            echo "  No Hugging Face token found - checked HF_TOKEN and your saved login."
            echo "  Do ONE of these, then re-run this script:"
            echo "      export HF_TOKEN=hf_your_token_here"
            echo "    - or - log in once:"
            echo "      $PY -c \"from huggingface_hub import login; login()\""
            echo "    - or - skip the gated model entirely with SMALL mode:"
            echo "      ./run_benchmark.sh $N_ROWS $LOAD_MODE small"
            exit 1
        fi
        echo "Hugging Face token detected."
    fi

    echo ""
    echo "=== [4/5] Model weights ==="
    echo "    The model downloads automatically on the first run (~16 GB). This can"
    echo "    take a while; later runs reuse the local cache."
fi

echo ""
echo "=== [5/5] Running benchmark ($N_ROWS rows, load mode: $LOAD_MODE) ==="
SMALL_FLAG=""
if [ -n "$TDA_SMALL" ]; then
    SMALL_FLAG="--small"
fi

"$PY" scripts/evaluate_humble_full_suite.py \
    --n "$N_ROWS" \
    --methods compact,humble_synthesis \
    --run-kind bench-standard \
    --oracle-cache-mode ignore_oracle \
    --load-mode "$LOAD_MODE" \
    --allow-downloads \
    $SMALL_FLAG \
    --output invariants/out/bench_bootstrap.json || exit 1

echo ""
echo "=== DONE ==="
echo "Results written to invariants/out/bench_bootstrap.json"
echo "Summarize anytime with:  $PY scripts/summarize_results.py"
