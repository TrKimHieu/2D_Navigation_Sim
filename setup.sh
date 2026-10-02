#!/usr/bin/env bash
# Install for Linux / macOS: creates .venv in the repository and installs hm3denv into it.
#
#   ./setup.sh              environments + CLI (numba kernels)
#   ./setup.sh --hub        + download of the pre-built datasets      (get-data)
#   ./setup.sh --build      + tools to build datasets from GLB files  (build-map, extend-data)
#   ./setup.sh --sim        + ZeroMQ server of the simulator          (sim --serve)
#   ./setup.sh --train      + PyTorch (CPU) and Stable-Baselines3     (train)
#   ./setup.sh --all        everything except --train, plus the tests
#   ./setup.sh --quiet      no "what next" text at the end
#
# Windows: setup.bat.
set -euo pipefail
cd "$(dirname "$0")"

extras="fast"
quiet=0
for arg in "$@"; do
  case "$arg" in
    --hub|--build|--sim|--train) extras="$extras,${arg#--}" ;;
    --dev|--all)   extras="$extras,hub,build,sim,test" ;;
    --extras=*)    extras="$extras,${arg#--extras=}" ;;
    -q|--quiet)    quiet=1 ;;
    -h|--help)     sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)             echo "unknown option: $arg (see ./setup.sh --help)" >&2; exit 2 ;;
  esac
done

fail() { echo "error: $1 failed. Check your internet connection and run ./setup.sh again." >&2; exit 3; }

if [ ! -x .venv/bin/python ]; then
  # Python 3.10-3.12 (newest first).
  py=""
  for cand in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1 &&
       "$cand" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 12))' 2>/dev/null; then
      py="$cand"; break
    fi
  done
  if [ -z "$py" ]; then
    echo "error: Python 3.10, 3.11 or 3.12 is required (found: $(python3 --version 2>&1 || echo none))." >&2
    echo "Install one (e.g. https://www.python.org/downloads/) and run ./setup.sh again." >&2
    exit 1
  fi
  echo "==> creating .venv with $("$py" --version)"
  "$py" -m venv .venv || fail "creating .venv"
  .venv/bin/python -m pip install --quiet --upgrade pip || fail "upgrading pip"
fi
py=.venv/bin/python

case ",$extras," in
  *,train,*)
    if ! "$py" -c "import torch" 2>/dev/null; then
      echo "==> installing PyTorch (CPU, once)"
      if [ "$(uname)" = Darwin ]; then
        "$py" -m pip install --quiet torch || fail "installing PyTorch"
      elif ! "$py" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu; then
        echo "==> the PyTorch CPU index (download.pytorch.org) is unreachable: installing torch from PyPI"
        echo "    instead. On Linux that wheel bundles the CUDA libraries (several GB on disk)."
        "$py" -m pip install --quiet torch || fail "installing PyTorch"
      fi
    fi ;;
esac

echo "==> installing hm3denv[$extras] (editable)"
"$py" -m pip install --quiet -e ".[$extras]" || fail "pip install"

echo "==> smoke test: oracle on the bundled demo dataset"
"$py" -c "from hm3denv.evaluate import evaluate
s = evaluate('demo-svg', robot='turtlebot4', agent='oracle', split='test', per_map=3, progress=False)
print(f\"    {s['episodes']} episodes, success {s['success']:.0%}, SPL {s['spl']:.2f}\")" || fail "smoke test"

[ "$quiet" = 1 ] && exit 0
cat <<'EOF'

Done. Every feature is one command (they install what they need on first use):

    ./app.sh                                menu of all features
    ./get-data.sh isb-svg-v1                download a dataset into data/datasets
    ./sim.sh --dataset demo-svg --view      open a simulation and watch it in the browser
    ./train.sh                              train PPO (demo data: a 5 minute check)
    ./build-map.sh path/to/scene.glb        turn GLB scenes into maps
    ./extend-data.sh NAME --status          continue a dataset

To use Python or the hm3d command yourself, activate the environment first:

    source .venv/bin/activate
EOF
