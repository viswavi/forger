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
# The evaluation protocol: 16,000-token answers at temperature 1.0,
# top-p 1.0 and no top-k, 5 per problem, with code also read from a reasoning
# trace the token limit cut off. The shared launcher passes coding's sampling
# settings first; these come after them, so they win, and flags given on the
# command line come later still.
export MAX_TOKENS="${MAX_TOKENS:-16000}"
PROTOCOL=(--temperature 1.0 --top-p 1.0 --top-k -1 --n-samples 5 --answer-extraction lenient)
# Statements reach ~7K tokens; keep Qwen3's full 40,960 context for longer answers.
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
exec bash "$TASK_DIR/../coding/eval_launcher.sh" "$TASK_DIR" "${PROTOCOL[@]}" "$@"
