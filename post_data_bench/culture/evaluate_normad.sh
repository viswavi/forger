#!/usr/bin/env bash
# Compatibility entry point; all evaluation uses the supported implementation.
exec bash "$(dirname "${BASH_SOURCE[0]}")/evaluate_model.sh" "$@"
