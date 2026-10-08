#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SLIME_ENV:-/home/vijayv/miniconda3/envs/slime}"

python launch_agent.py "$@"
