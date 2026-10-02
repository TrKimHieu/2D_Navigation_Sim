#!/usr/bin/env bash
# Menu of every hm3denv feature (double-click me).
# Usage: ./app.sh --help
exec "$(dirname "$0")/scripts/run.sh" - scripts/app.py "$@"
