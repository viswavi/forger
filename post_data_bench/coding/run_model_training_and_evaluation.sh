#!/usr/bin/env bash
# Train and evaluate an existing generated_raw_trl_format.jsonl.
# Usage: sbatch run_model_training_and_evaluation.sh --agent-data-dir <run-dir>
#        [--model Qwen/Qwen3-4B] [--split test] [--no-thinking]
#
# --no-thinking derives bare-answer targets from the existing converted file,
# evaluates without thinking, and uses a distinct output suffix and 4096-token
# default limit. --max-length overrides that limit; match the conversion
# filter to avoid truncated training targets. Longer sequences use more memory.
# --grad-accum overrides accumulation (GPUs × accumulation gives batch size).
# --lr is scaled 10x for LoRA; targets default to q_proj,v_proj.

#SBATCH --job-name=train_eval_coding
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=90G
#SBATCH --time=16:00:00
#SBATCH --output=train_eval_model_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$SCRIPT_DIR/prompt_building.py" ]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR:?Set SLURM_SUBMIT_DIR to the task source directory}"
fi
[ -f "$SCRIPT_DIR/prompt_building.py" ] || { echo "Task source directory not found" >&2; exit 1; }
cd "$SCRIPT_DIR"

CONDA_ENV="trl"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

MODEL_OUTPUT_TASK_DIR="/data/user_data/vijayv/tir5/agent_data_models/coding"

MODEL="Qwen/Qwen3-4B"
SPLIT="val"
AGENT_DATA_DIR=""
GRAD_ACCUM=""
MAX_LENGTH=""
LR=""
LORA_TARGET_MODULES="q_proj,v_proj"
NO_THINKING=0
FULL_FINETUNE=0
ASSISTANT_ONLY_LOSS=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent-data-dir) AGENT_DATA_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --grad-accum) GRAD_ACCUM="$2"; shift 2 ;;
    --max-length) MAX_LENGTH="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --lora-target-modules) LORA_TARGET_MODULES="$2"; shift 2 ;;
    --no-thinking) NO_THINKING=1; shift ;;
    --full-finetune) FULL_FINETUNE=1; shift ;;
    --assistant-only-loss) ASSISTANT_ONLY_LOSS=1; shift ;;
    --eval-arg) EXTRA_ARGS+=("$2"); shift 2 ;;
    *) echo "Unknown argument: $1 (use --eval-arg for evaluation options)" >&2; exit 2 ;;
  esac
done

if [ -z "$AGENT_DATA_DIR" ]; then
  echo "!!! usage: sbatch $0 --agent-data-dir <path> [--model <model>] [--split val|test] [--no-thinking]"
  exit 1
fi

# No-thinking targets fit a 4096-token default, reducing activation memory.
# Explicit --max-length overrides it; thinking mode uses train_model’s default.
if [[ "$NO_THINKING" -eq 1 && -z "$MAX_LENGTH" ]]; then
  MAX_LENGTH="4096"
  echo ">>> --no-thinking with no --max-length given: defaulting max-length -> $MAX_LENGTH"
fi

THINKING_DATA_FILE="${AGENT_DATA_DIR%/}/generated_raw_trl_format.jsonl"
if [ ! -f "$THINKING_DATA_FILE" ]; then
  echo "!!! Missing training data: $THINKING_DATA_FILE" >&2
  exit 1
fi

if [ "$NO_THINKING" -eq 1 ]; then
  DATA_FILE="${AGENT_DATA_DIR%/}/generated_raw_trl_format_no_thinking.jsonl"
  if [ ! -f "$DATA_FILE" ] || [ "$THINKING_DATA_FILE" -nt "$DATA_FILE" ]; then
    echo ">>> $DATA_FILE not found; deriving it from $THINKING_DATA_FILE via strip_thinking_from_trl.py"
    python strip_thinking_from_trl.py "$THINKING_DATA_FILE"
  fi
else
  DATA_FILE="$THINKING_DATA_FILE"
fi

# Reuse the generation run tag to keep checkpoints traceable to their data.
RUN_TAG="$(basename "${AGENT_DATA_DIR%/}" | sed -n 's/^agent_run_//p')"
[ -z "$RUN_TAG" ] && RUN_TAG="$(date +%Y%m%d_%H%M%S)_${RANDOM}"

MODEL_TAG="$(echo "$MODEL" | tr '/:' '__')"
LR_TAG=""
[[ -n "$LR" ]] && LR_TAG="_lr${LR}"
THINK_TAG=""
[[ "$NO_THINKING" -eq 1 ]] && THINK_TAG="_no_thinking"
FULL_FT_TAG=""
[[ "$FULL_FINETUNE" -eq 1 ]] && FULL_FT_TAG="_fullft"
AOL_TAG=""
[[ "$ASSISTANT_ONLY_LOSS" -eq 1 ]] && AOL_TAG="_assistonly"
OUTPUT_DIR="${MODEL_OUTPUT_TASK_DIR}/${MODEL_TAG}_${RUN_TAG}${LR_TAG}${THINK_TAG}${FULL_FT_TAG}${AOL_TAG}_$(date +%Y%m%d_%H%M%S)_${RANDOM}"

TRAIN_ARGS=()
if [[ -n "$GRAD_ACCUM" ]]; then
  TRAIN_ARGS+=(--train-arg --grad-accum --train-arg "$GRAD_ACCUM")
  ACTUAL_GPUS="${SLURM_GPUS_ON_NODE:-}"
  if [[ -z "$ACTUAL_GPUS" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then IFS=, read -ra devices <<< "$CUDA_VISIBLE_DEVICES"; ACTUAL_GPUS="${#devices[@]}"; else ACTUAL_GPUS="$(nvidia-smi -L | wc -l)"; fi
  fi
  echo ">>> overriding grad-accum -> $GRAD_ACCUM (effective batch = ${ACTUAL_GPUS} x ${GRAD_ACCUM} = $((ACTUAL_GPUS * GRAD_ACCUM)))"
fi
if [[ -n "$MAX_LENGTH" ]]; then
  TRAIN_ARGS+=(--max-length "$MAX_LENGTH")
  echo ">>> overriding max-length -> $MAX_LENGTH (must match the cutoff generated_raw_trl_format.jsonl was converted at)"
fi
if [[ "$FULL_FINETUNE" -eq 1 ]]; then
  TRAIN_ARGS+=(--full-finetune)
  echo ">>> full-parameter fine-tuning (no LoRA); --lora-target-modules is ignored"
  if [[ -n "$LR" ]]; then
    TRAIN_ARGS+=(--lr "$LR")
    echo ">>> overriding lr -> $LR (used as-is -- full-finetune runs are NOT auto-scaled 10x)"
  fi
else
  if [[ -n "$LR" ]]; then
    TRAIN_ARGS+=(--lr "$LR")
    echo ">>> overriding lr -> $LR (effective, post 10x LoRA scaling: $(python3 -c "print(float('$LR')*10)"))"
  fi
  TRAIN_ARGS+=(--train-arg --lora-target-modules --train-arg "$LORA_TARGET_MODULES")
  echo ">>> lora-target-modules -> $LORA_TARGET_MODULES"
fi
if [[ "$ASSISTANT_ONLY_LOSS" -eq 1 ]]; then
  TRAIN_ARGS+=(--train-arg --assistant-only-loss)
  echo ">>> assistant-only loss: masking the prompt out of the loss (default is the full sequence)"
fi

./train_model.sh --model "$MODEL" --data-file "$DATA_FILE" --output-dir "$OUTPUT_DIR" "${TRAIN_ARGS[@]}"

EVAL_ARGS=("${EXTRA_ARGS[@]}")
[[ "$NO_THINKING" -eq 1 ]] && EVAL_ARGS+=(--no-thinking)
# A local checkpoint: serve it with vLLM unless SERVER_BACKEND or --eval-arg says otherwise.
./evaluate_model.sh --server-backend "${SERVER_BACKEND:-vllm}" --split "$SPLIT" --model-path "$OUTPUT_DIR" "${EVAL_ARGS[@]}"
