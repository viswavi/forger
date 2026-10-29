#!/bin/bash
# Build deterministic FrontierCS validation/secret-test splits (problems + tests).
# Existing test outputs require --force; --output-dir creates a new version.
set -e
cd "$(dirname "$0")"

echo ">>> [1/1] validation_tasks.jsonl / .hidden/test_tasks.jsonl (make_splits.py)"
python3 make_splits.py "$@"

echo ">>> done."
