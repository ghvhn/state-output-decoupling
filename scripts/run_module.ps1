param(
    [Parameter(Mandatory = $true)]
    [string] $Module,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ModuleArgs
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Missing local Python environment at $Python"
}

$env:PYTHONPATH = [string] $Root
Push-Location $Root
try {
    & $Python -u -m $Module @ModuleArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
