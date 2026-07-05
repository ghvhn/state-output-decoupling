#!/usr/bin/env bash
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
#  Optional first argument: a model id/path to run the egg shell on, e.g.
#      ./run_phenomenality_shell.sh Qwen/Qwen2.5-1.5B-Instruct
#  When launched from the benchmark egg, the model is passed via EGG_MODEL.
.venv/bin/python scripts/interactive_phenomenality.py "$@"
echo ""
echo "[Phenomenality shell exited. Press Enter to close.]"
read -r
