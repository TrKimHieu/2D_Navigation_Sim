#!/usr/bin/env bash
# Continue a dataset: new scenes, more tasks on its maps, new robots.
# Usage: ./extend-data.sh --help
exec "$(dirname "$0")/scripts/run.sh" build,hub hm3denv extend "$@"
