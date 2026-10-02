# Install for Windows: creates .venv in the repository and installs hm3denv into it.
# Called by .\setup.bat (and by run.ps1 when a feature needs more packages).
#
#   .\setup.bat              environments + CLI (numba kernels)
#   .\setup.bat --hub        + download of the pre-built datasets      (get-data)
#   .\setup.bat --build      + tools to build datasets from GLB files  (build-map, extend-data)
#   .\setup.bat --sim        + ZeroMQ server of the simulator          (sim --serve)
#   .\setup.bat --train      + PyTorch (CPU) and Stable-Baselines3     (train)
#   .\setup.bat --all        everything except --train, plus the tests
#   .\setup.bat --quiet      no "what next" text at the end
#
# Options can also be written -Hub, -Build, ... or --extras=hub,build.
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $root

$extras = [System.Collections.Generic.List[string]]@("fast")
$quiet = $false
foreach ($a in $args) {
    switch -Regex ($a) {
        '^--?hub$'          { $extras.Add("hub") }
        '^--?build$'        { $extras.Add("build") }
        '^--?sim$'          { $extras.Add("sim") }
        '^--?train$'        { $extras.Add("train") }
        '^--?(all|dev)$'    { "hub", "build", "sim", "test" | ForEach-Object { $extras.Add($_) } }
        '^--?extras=(.*)$'  { $Matches[1] -split "," | Where-Object { $_ } | ForEach-Object { $extras.Add($_) } }
        '^--?(q|quiet)$'    { $quiet = $true }
        '^--?(h|help)$'     { Get-Content $PSCommandPath | Select-Object -Skip 2 -First 10 | ForEach-Object { $_ -replace '^# ?', '' }; exit 0 }
        default             { Write-Host "unknown option: $a (see .\setup.bat --help)" -ForegroundColor Red; exit 2 }
    }
}
$extras = @($extras | Select-Object -Unique)

function Assert-Ok($what) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "error: $what failed (exit code $LASTEXITCODE). Check your internet connection and run .\setup.bat again." -ForegroundColor Red
        exit 3
    }
}

# Python 3.10-3.12 (newest first): the py launcher, then python / python3 on PATH.
$check = "import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 12))"
$py = $null
foreach ($c in "py -3.12", "py -3.11", "py -3.10", "python", "python3") {
    $cand = @($c -split " ")
    if (-not (Get-Command $cand[0] -ErrorAction SilentlyContinue)) { continue }
    $pre = @($cand | Select-Object -Skip 1)
    try { & $cand[0] @pre -c $check 2>$null } catch { continue }   # e.g. "No suitable Python runtime"
    if ($LASTEXITCODE -eq 0) { $py = $cand; break }
}

$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    if (-not $py) {
        Write-Host ("error: Python 3.10, 3.11 or 3.12 is required. Install one from https://www.python.org/downloads/ " +
                    "(tick 'Add python.exe to PATH') and run .\setup.bat again.") -ForegroundColor Red
        exit 1
    }
    Write-Host "==> creating .venv"
    $pre = @($py | Select-Object -Skip 1)
    & $py[0] @pre -m venv .venv; Assert-Ok "creating .venv"
    & $venvPy -m pip install --quiet --upgrade pip; Assert-Ok "upgrading pip"
}

function Test-Import($module) {
    # stderr of a native command is a terminating error under "Stop" in PowerShell 5.1
    try { & $venvPy -c "import $module" 2>$null; return ($LASTEXITCODE -eq 0) } catch { return $false }
}

if ($extras -contains "train") {
    if (-not (Test-Import "torch")) {
        Write-Host "==> installing PyTorch (CPU, once)"
        & $venvPy -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
        if ($LASTEXITCODE -ne 0) {
            Write-Host "==> the PyTorch CPU index (download.pytorch.org) is unreachable: installing torch from PyPI instead"
            & $venvPy -m pip install --quiet torch
        }
        Assert-Ok "installing PyTorch"
    }
}

$list = $extras -join ","
Write-Host "==> installing hm3denv[$list] (editable)"
& $venvPy -m pip install --quiet -e ".[$list]"; Assert-Ok "pip install"

Write-Host "==> smoke test: oracle on the bundled demo dataset"
# No double quotes inside: Windows PowerShell 5.1 mangles them in native arguments.
$smoke = @"
from hm3denv.evaluate import evaluate
s = evaluate('demo-svg', robot='turtlebot4', agent='oracle', split='test', per_map=3, progress=False)
print('    %d episodes, success %.0f %%, SPL %.2f' % (s['episodes'], 100 * s['success'], s['spl']))
"@
& $venvPy -c $smoke; Assert-Ok "smoke test"
if ($quiet) { exit 0 }

Write-Host @"

Done. Every feature is one command (they install what they need on first use):

    .\app.bat                               menu of all features
    .\get-data.bat isb-svg-v1               download a dataset into data\datasets
    .\sim.bat --dataset demo-svg --view     open a simulation and watch it in the browser
    .\train.bat                             train PPO (demo data: a 5 minute check)
    .\build-map.bat path\to\scene.glb       turn GLB scenes into maps
    .\extend-data.bat NAME --status         continue a dataset

To use Python or the hm3d command yourself, activate the environment first:

    .venv\Scripts\activate.bat            (PowerShell: .venv\Scripts\Activate.ps1)
"@
