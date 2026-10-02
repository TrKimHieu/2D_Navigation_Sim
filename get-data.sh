#!/usr/bin/env bash
# Download pre-built datasets from Hugging Face into data/datasets.
# Usage: ./get-data.sh --help
exec "$(dirname "$0")/scripts/run.sh" hub scripts/get_data.py "$@"
