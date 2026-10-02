#!/usr/bin/env bash
# Open a simulation (dataset, map, robot, number of envs, start/goal); watch it or serve it over ZeroMQ.
# Usage: ./sim.sh --help
exec "$(dirname "$0")/scripts/run.sh" sim hm3denv sim "$@"
