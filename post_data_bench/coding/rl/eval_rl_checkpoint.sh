#!/usr/bin/env bash
# Evaluate one RL checkpoint on one GPU using a strided test subset.
# Usage: sbatch rl/eval_rl_checkpoint.sh [--checkpoint hf_rollout39]
#        [--model-path <HF-dir>] [--num-shards 13]
#
# Default: latest checkpoint, 1/9 of test rows. Striding avoids prefix bias;
# more shards reduce the subset. KV-cache limits throughput, so response
# and context caps are tuned for one GPU. Saved predictions resume across
# submissions; the default subset may need two walltime allocations.
# Add GPU constraints at submission time if needed.
#SBATCH --job-name=lcb-eval-rl-ckpt
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=90G
#SBATCH --time=1:45:00
#SBATCH --output=logs/lcb-eval-rl-ckpt-%j.out
#SBATCH --error=logs/lcb-eval-rl-ckpt-%j.err

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$SCRIPT_DIR/evaluate_model.py" ] || SCRIPT_DIR="${SLURM_SUBMIT_DIR:?Submit from coding/}"
[ -f "$SCRIPT_DIR/evaluate_model.py" ] || exit 1
cd "$SCRIPT_DIR"

RL_RUN_GLOB="/data/user_data/vijayv/tir5/rl_runs/coding/GRPO_*balanced_204_20260907_210046"
CHECKPOINT=""
MODEL_PATH=""
NUM_SHARDS=9
SHARD_INDEX=0
SPLIT=val

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)  CHECKPOINT="$2"; shift 2 ;;
    --model-path)  MODEL_PATH="$2"; shift 2 ;;
    --num-shards)  NUM_SHARDS="$2"; shift 2 ;;
    --shard-index) SHARD_INDEX="$2"; shift 2 ;;
    --split)       SPLIT="$2"; shift 2 ;;
    *) echo "!!! unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$MODEL_PATH" ]; then
  RUN=$(ls -d $RL_RUN_GLOB 2>/dev/null | head -1)
  [ -n "$RUN" ] || { echo "!!! no RL run matching $RL_RUN_GLOB" >&2; exit 1; }
  if [ -z "$CHECKPOINT" ]; then
    # Highest hf_rollout<N>, numerically -- plain glob order puts 9 after 39.
    CHECKPOINT=$(ls -d "$RUN"/hf_rollout* 2>/dev/null \
                 | sed 's/.*hf_rollout//' | sort -n | tail -1 | sed 's/^/hf_rollout/')
  fi
  MODEL_PATH="$RUN/$CHECKPOINT"
fi

[ -e "$MODEL_PATH/config.json" ] || { echo "!!! not an HF checkpoint: $MODEL_PATH" >&2; exit 1; }

# Partial diagnostics have their own directory and never write full-benchmark metrics.
ESC_MODEL="$(echo "$MODEL_PATH" | sed 's#/#__#g')"
PARTIAL_DIR="model_outputs/${ESC_MODEL}/${SPLIT}_stride${NUM_SHARDS}_${SHARD_INDEX}"
SHARD_DIR="$PARTIAL_DIR/shard_${SHARD_INDEX}"

echo ">>> checkpoint : $MODEL_PATH"
echo ">>> split      : $SPLIT, shard $SHARD_INDEX of $NUM_SHARDS"
echo ">>> out dir    : $SHARD_DIR"
echo ">>> resumable  : rows already in $SHARD_DIR/predictions.jsonl are skipped"

# See the sizing note above: ctx and token cap are the throughput levers, and
# MAX_NUM_SEQS is set to what the KV cache can actually hold rather than 128.
export MAX_MODEL_LEN=24576
export MAX_TOKENS=20480
export MAX_NUM_SEQS=12
export MAX_CONN=12
export GPU_MEM=0.90

# A local HF checkpoint: serve it with vLLM unless SERVER_BACKEND says otherwise.
./evaluate_model.sh \
  --server-backend "${SERVER_BACKEND:-vllm}" \
  --no-merge \
  --out-dir "$PARTIAL_DIR" \
  --split "$SPLIT" \
  --model-path "$MODEL_PATH" \
  --num-shards "$NUM_SHARDS" \
  --shard-indices "$SHARD_INDEX"
rc=$?

PRED="$SHARD_DIR/predictions.jsonl"
[ -f "$PRED" ] || PRED="model_outputs/${ESC_MODEL}/${SPLIT}/predictions.jsonl"
echo
echo ">>> evaluate_model.sh exit=$rc; summarising $PRED"
# summarize_eval.py is stdlib-only, so any interpreter works; `python` may not
# exist outside a conda env even where `python3` does.
PY_BIN=$(command -v python || command -v python3)
"$PY_BIN" rl/summarize_eval.py "$PRED"

exit "$rc"
