#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$TASK_DIR/prompt_building.py" ]]; then
  TASK_DIR="${SLURM_SUBMIT_DIR:?Submit from the task source directory}"
fi
[[ -f "$TASK_DIR/prompt_building.py" ]] || exit 1
# The shared launcher defaults to Tinker, which this evaluator cannot sample from yet:
# keep the previous default (the pinned backend, else vLLM). --server-backend overrides.
export SERVER_BACKEND="${SERVER_BACKEND:-${PDB_EVAL_BACKEND:-vllm}}"
# Statements reach ~7K tokens: 28,672-token answers need Qwen3's full 40,960 context.
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
exec bash "$TASK_DIR/../coding/eval_launcher.sh" "$TASK_DIR" "$@"
