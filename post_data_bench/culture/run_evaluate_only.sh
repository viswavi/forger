#!/usr/bin/env bash
# Evaluate an existing checkpoint.
# Usage: sbatch run_evaluate_only.sh --model-path <checkpoint> [--split test]
# Qwen3 --no-thinking disables reasoning and adds an output-directory suffix.
#SBATCH --job-name=eval_culture
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=90G
#SBATCH --time=04:00:00
#SBATCH --output=eval_model_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$SCRIPT_DIR/prompt_building.py" ]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR:?Set SLURM_SUBMIT_DIR to the task source directory}"
fi
[ -f "$SCRIPT_DIR/prompt_building.py" ] || { echo "Task source directory not found" >&2; exit 1; }
cd "$SCRIPT_DIR"

CONDA_ENV="${EVAL_CONDA_ENV:-slime}"
export SERVER_BACKEND="${SERVER_BACKEND:-sglang}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

SPLIT="test"
MODEL_PATH=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [ -z "$MODEL_PATH" ]; then
  echo "!!! usage: sbatch $0 --model-path <path> [--split val|test] [--no-thinking]"
  exit 1
fi

./evaluate_model.sh --split "$SPLIT" --model-path "$MODEL_PATH" "${EXTRA_ARGS[@]}"
