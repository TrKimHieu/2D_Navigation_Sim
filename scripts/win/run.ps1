# Runs one hm3denv feature inside .venv, installing what it needs first. Used by the .bat
# files at the root of the repository:
#
#   run.ps1 <extras> <target> [arguments ...]
#     extras   comma-separated setup extras the feature needs (hub, build, sim, train) or "-"
#     target   a script (scripts\get_data.py) or a module (hm3denv) run with python -m
#
# Exit codes: the feature's own, 1 no Python, 3 installation failed.
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $root

$extras, $target, $rest = $args
$rest = @($rest | Where-Object { $null -ne $_ })
$need = @(if ($extras -and $extras -ne "-") { $extras -split "," | Where-Object { $_ } })

# modules that show an extra is installed (hm3denv itself is always needed)
$probe = @{ hub = "huggingface_hub"; build = "trimesh,shapely,PIL"; sim = "zmq";
            train = "torch,stable_baselines3"; fast = "numba"; test = "pytest" }
$mods = @("hm3denv") + @($need | ForEach-Object { $probe[$_] -split "," })

$venvPy = Join-Path $root ".venv\Scripts\python.exe"
$ok = $false
if (Test-Path $venvPy) {
    $code = "import importlib.util, sys; sys.exit(any(importlib.util.find_spec(m) is None for m in sys.argv[1:]))"
    # stderr of a native command is a terminating error under "Stop" in PowerShell 5.1
    try { & $venvPy -c $code @mods 2>$null; $ok = ($LASTEXITCODE -eq 0) } catch { $ok = $false }
}
if (-not $ok) {
    $setupArgs = @("--quiet") + @($need | ForEach-Object { "--$_" })
    & (Join-Path $PSScriptRoot "setup.ps1") @setupArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($target -like "*.py") {
    & $venvPy $target @rest
} else {
    & $venvPy -m $target @rest
}
exit $LASTEXITCODE
