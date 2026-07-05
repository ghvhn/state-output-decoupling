#!/usr/bin/env bash
# ============================================================================
#  Run this file to start the benchmark. No coding needed.
#
#  It runs the small, open Qwen model on your CPU -- no GPU, no Hugging Face
#  account, no license. The first run downloads ~3 GB and then scores a few
#  rows. Later you can run the full Llama model with run_benchmark.sh.
# ============================================================================
cd "$(dirname "$0")"

echo ""
echo "===================================================================="
echo "  TDA benchmark - easy start (small model, CPU, no setup needed)"
echo "===================================================================="
echo ""

echo "Checking for Python..."
if ! command -v python3 &> /dev/null; then
    echo ""
    echo "  Python 3 is not installed, or it is not on your PATH."
    echo ""
    echo "  Please install Python 3.11 or newer and try again."
    echo ""
    exit 1
fi
echo "Found Python."
echo ""
echo "This will now set things up and run a 3-row benchmark on the CPU."
echo "The first run downloads about 3 GB, so it can take a while. That's normal."
echo ""
read -p "Press Enter to continue..."

bash run_benchmark.sh 3 cpu small

echo ""
echo "===================================================================="
echo "  Finished. Results are in invariants/out/bench_bootstrap.json"
echo "===================================================================="
