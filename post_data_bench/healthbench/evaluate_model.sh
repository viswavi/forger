#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$TASK_DIR/prompt_building.py" ]]; then
  TASK_DIR="${SLURM_SUBMIT_DIR:?Submit from the task source directory}"
fi
[[ -f "$TASK_DIR/prompt_building.py" ]] || exit 1
# The shared launcher defaults to Tinker, which this evaluator cannot sample from:
# use the pinned backend, else vLLM. --server-backend overrides.
export SERVER_BACKEND="${SERVER_BACKEND:-${PDB_EVAL_BACKEND:-vllm}}"
# The evaluation protocol: up to 28,672 generated tokens at temperature 1.0,
# top-p 1.0 and no top-k, 8 answers per chat. The shared launcher passes coding's
# sampling settings first; these come after them, so they win, and flags given on
# the command line come later still.
export MAX_TOKENS="${MAX_TOKENS:-28672}"
PROTOCOL=(--temperature 1.0 --top-p 1.0 --top-k -1 --n-samples 8)
# Chats reach ~8K tokens; keep Qwen3's full 40,960 context for long answers.
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
# The agent's copy has the shared launcher beside this file; the host's is in coding/.
LAUNCHER="$TASK_DIR/eval_launcher.sh"
[[ -f "$LAUNCHER" ]] || LAUNCHER="$TASK_DIR/../coding/eval_launcher.sh"
exec bash "$LAUNCHER" "$TASK_DIR" "${PROTOCOL[@]}" "$@"
