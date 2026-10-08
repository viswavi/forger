#!/bin/bash
# Print step reward curves and write <run-dir>/curves.csv.
# Usage: rl/curves.sh [run-dir] [--bucket 256]  # defaults to newest run
# Safe during live jobs; partial steps overrepresent short abstentions.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROOT="${RL_OUTPUT_ROOT:-/data/user_data/vijayv/tir5/rl_runs/coding}"
RUN="${1:-}"
if [ -z "$RUN" ] || [[ "$RUN" == --* ]]; then
  RUN="$(ls -dt "$ROOT"/GRPO_*/ 2>/dev/null | head -1)"
  [ -n "$RUN" ] || { echo "!!! no GRPO_* run under $ROOT" >&2; exit 1; }
  RUN="${RUN%/}"
  echo ">>> newest run: $RUN"
else
  shift
fi
[ -d "$RUN/logs/rollouts" ] || { echo "!!! no $RUN/logs/rollouts" >&2; exit 1; }

# The step boundaries come from the job's stdout, so find the log whose output
# directory matches this run rather than assuming a job name.
TRAIN_LOG=""
for f in logs/*.out; do
  [ -f "$f" ] || continue
  if grep -qF ">>> out=$RUN" "$f" 2>/dev/null; then TRAIN_LOG="$f"; break; fi
done

ARGS=("$RUN/logs/rollouts" --csv "$RUN/curves.csv")
if [ -n "$TRAIN_LOG" ]; then
  echo ">>> step boundaries from $TRAIN_LOG"
  ARGS+=(--train-log "$TRAIN_LOG")
else
  echo ">>> no matching .out found; falling back to gap segmentation"
fi
PYTHON_BIN="$(command -v python || command -v python3)"
exec "$PYTHON_BIN" rl/summarize_rollouts.py "${ARGS[@]}" "$@"
