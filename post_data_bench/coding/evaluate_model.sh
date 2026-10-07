#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$TASK_DIR/prompt_building.py" ]]; then
  TASK_DIR="${SLURM_SUBMIT_DIR:?Submit from the task source directory}"
fi
[[ -f "$TASK_DIR/prompt_building.py" ]] || exit 1
exec bash "$TASK_DIR/eval_launcher.sh" "$TASK_DIR" "$@"
