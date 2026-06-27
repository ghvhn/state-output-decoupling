@echo off
setlocal
cd /d "%~dp0\.."

".venv\Scripts\python.exe" -u scripts\evaluate_humble_full_suite.py ^
  --n all ^
  --methods all ^
  --max-rounds 2 ^
  --required-agreement 2 ^
  --max-new-tokens 100 ^
  --repair-token-multiplier 3 ^
  --max-attempt-tokens 300 ^
  --max-elapsed-sec 180 ^
  --load-mode auto ^
  --resume ^
  --output invariants\out\humble_full_suite_gsm8k_all.json ^
  > invariants\out\humble_full_suite_gsm8k_all.log ^
  2> invariants\out\humble_full_suite_gsm8k_all.err.log
