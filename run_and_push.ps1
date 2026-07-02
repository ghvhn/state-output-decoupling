$env:PYTHONPATH="."
Write-Host "Running Benchmark..."
python scripts\evaluate_humble_full_suite.py --hard-only --run-kind bench-standard --oracle-cache-mode ignore_oracle --base-max-time-sec 60 --max-synthesis-events 1 --max-synthesis-steps 24 --oracle-curriculum off --output invariants\out\humble_full_suite_gsm8k_standard_fresh.json

if ($env:PUSH_AFTER_RUN -eq "1") {
    Write-Host "Benchmark Completed! Pushing Cognitive Cache to GitHub..."
    git add -f invariants/data/cognitive_cache.pt
    git add -f invariants/out/humble_full_suite_gsm8k_standard_fresh.json
    git commit -m "Auto-Update: Pushed cognitive cache and benchmark results post-run"
    git push
} else {
    Write-Host "Benchmark Completed. PUSH_AFTER_RUN is not 1, so nothing was committed or pushed."
}

Write-Host "All Done!"
