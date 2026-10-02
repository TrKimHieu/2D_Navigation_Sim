#!/usr/bin/env bash
# Runs one hm3denv feature inside .venv, installing what it needs first. Used by the .sh
# files at the root of the repository:
#
#   scripts/run.sh <extras> <target> [arguments ...]
#     extras   comma-separated setup extras the feature needs (hub, build, sim, train) or "-"
#     target   a script (scripts/get_data.py) or a module (hm3denv) run with python -m
#
# Exit codes: the feature's own, 1 no Python, 3 installation failed.
set -euo pipefail
cd "$(dirname "$0")/.."

extras="$1"; target="$2"; shift 2
need=()
[ "$extras" != "-" ] && IFS=',' read -r -a need <<< "$extras"

mods=(hm3denv)                       # modules that show an extra is installed
for e in ${need[@]+"${need[@]}"}; do
  case "$e" in
    hub)   mods+=(huggingface_hub) ;;
    build) mods+=(trimesh shapely PIL) ;;
    sim)   mods+=(zmq) ;;
    train) mods+=(torch stable_baselines3) ;;
    fast)  mods+=(numba) ;;
    test)  mods+=(pytest) ;;
  esac
done

py=.venv/bin/python
if [ ! -x "$py" ] || ! "$py" -c 'import importlib.util, sys
sys.exit(any(importlib.util.find_spec(m) is None for m in sys.argv[1:]))' "${mods[@]}" 2>/dev/null; then
  args=(--quiet)
  for e in ${need[@]+"${need[@]}"}; do args+=("--$e"); done
  ./setup.sh "${args[@]}"
fi

case "$target" in
  *.py) exec "$py" "$target" "$@" ;;
  *)    exec "$py" -m "$target" "$@" ;;
esac
