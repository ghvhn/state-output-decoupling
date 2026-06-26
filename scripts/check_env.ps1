$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Missing local Python environment at $Python"
}

& $Python (Join-Path $Root "scripts\check_env.py")

exit $LASTEXITCODE
