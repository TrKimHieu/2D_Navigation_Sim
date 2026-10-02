#!/usr/bin/env bash
# Turn .glb scenes into navigation maps and tasks (a dataset in data/datasets).
# Usage: ./build-map.sh --help
exec "$(dirname "$0")/scripts/run.sh" build hm3denv build-map "$@"
