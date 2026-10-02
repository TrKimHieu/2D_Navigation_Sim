#!/usr/bin/env bash
# Train a PPO agent (Stable-Baselines3) and evaluate it on unseen maps.
# Usage: ./train.sh --help
exec "$(dirname "$0")/scripts/run.sh" train examples/train_ppo.py "$@"
