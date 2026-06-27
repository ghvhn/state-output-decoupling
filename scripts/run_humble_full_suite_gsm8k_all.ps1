$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$venv = Join-Path $root ".venv"
$sitePackages = Join-Path $venv "Lib\site-packages"
$python = "C:\Users\Gavin Powell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$log = Join-Path $root "invariants\out\humble_full_suite_gsm8k_all.log"
$err = Join-Path $root "invariants\out\humble_full_suite_gsm8k_all.err.log"

$env:PYTHONPATH = "$venv;$sitePackages;$env:PYTHONPATH"
$env:VIRTUAL_ENV = $venv
$env:PATH = (Join-Path $venv "Scripts") + ";$env:PATH"

& $python -u scripts\evaluate_humble_full_suite.py `
  --n all `
  --methods all `
  --max-rounds 2 `
  --required-agreement 2 `
  --max-new-tokens 100 `
  --repair-token-multiplier 3 `
  --max-attempt-tokens 300 `
  --max-elapsed-sec 180 `
  --load-mode auto `
  --resume `
  --output invariants\out\humble_full_suite_gsm8k_all.json `
  1> $log `
  2> $err
