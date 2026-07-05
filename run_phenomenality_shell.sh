#!/usr/bin/env bash
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8

PY=".venv/bin/python"

echo ""
echo "=== [1/3] Python virtual environment ==="
if [ ! -f "$PY" ]; then
    echo "Creating .venv ..."
    python3 -m venv .venv || exit 1
else
    echo "Reusing existing .venv"
fi

echo ""
echo "=== [2/3] Installing dependencies ==="
"$PY" -m pip install --upgrade pip >/dev/null
"$PY" -m pip install -r requirements-bench.txt || exit 1

echo ""
echo "=== [3/3] Checking Hugging Face authentication ==="
if [ -n "$HF_TOKEN" ]; then
    echo "Using HF_TOKEN from environment."
else
    "$PY" -c "import sys; from huggingface_hub import get_token; sys.exit(0 if get_token() else 1)" >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo ""
        echo "  No Hugging Face token found. The shell may fall back to smaller open models"
        echo "  or fail if you explicitly request a gated model without logging in."
        echo "  To log in once: $PY -c \"from huggingface_hub import login; login()\""
        echo ""
    else
        echo "Hugging Face token detected."
    fi
fi

echo ""
echo "=== Launching Phenomenality Shell ==="
#  Optional first argument: a model id/path to run the egg shell on, e.g.
#      ./run_phenomenality_shell.sh Qwen/Qwen2.5-1.5B-Instruct
"$PY" scripts/interactive_phenomenality.py "$@"

echo ""
echo "[Phenomenality shell exited. Press Enter to close.]"
read -r
