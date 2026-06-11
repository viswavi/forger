#!/bin/bash
# =============================================================================
# train_and_eval_gsm8k_basic.sh -- single-stage SFT on the GSM8K "basic" TRL
# dataset (examples/gsm8k/gsm8k_basic_trl.jsonl), then evaluate on GSM8K.
#
# Stripped-down pipeline: base model -> ONE SFT stage on the GSM8K dataset -> eval.
# No Capybara warm-up, no Task-SFT, no three-stage chaining.
#
# Evaluated models:
#   base         the untrained base model (baseline; skip with --no-baseline)
#   base+gsm8k   base SFT'd on the GSM8K basic dataset
#
# Trains with LoRA adapters (--use-lora); train_sft.py auto-scales the LR 10x and merges
# the adapter back into a full model so the eval is unchanged.
#
# Output run dir: .../forger_training_runs_gsm8k_high_lr_lora/basic/<model>
# =============================================================================
#SBATCH --job-name=sft_eval_gsm8k_basic
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=90G
#SBATCH --time=12:00:00
#SBATCH --output=train_and_eval_gsm8k_basic_%j.log

set -euo pipefail

# ---- variant (the ONLY difference vs train_and_eval_gsm8k_full.sh) ----------
VARIANT="basic"
DATA_FILE_NAME="gsm8k_basic_trl.jsonl"

# ---- defaults ---------------------------------------------------------------
MODEL="Qwen/Qwen3-1.7B-Base"
CONDA_ENV="trl"
EXAMPLES_DIR="/home/vijayv/forger/examples"
PTB_DIR="/home/vijayv/PostTrainBench"
RUNS_DIR=""
# Gentle recipe (see train_and_eval_capybara_15k.sh): a low LR keeps the SFT close to the
# robust base model. Override with --lr / --epochs.
LR="2e-6"
EPOCHS="6"
EVAL_LIMIT="-1"           # -1 = full benchmark; set small for quick checks
GPU_MEM_UTIL="0.6"        # vLLM gpu_memory_utilization for the eval phase
RUN_BASELINE=1            # also eval the untrained base
SMOKE=0                   # cheap end-to-end check
EXTRA_TRAIN_ARGS=()

# ---- arg parsing ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)        MODEL="$2"; shift 2;;
    --epochs)       EPOCHS="$2"; shift 2;;
    --lr)           LR="$2"; shift 2;;
    --eval-limit)   EVAL_LIMIT="$2"; shift 2;;
    --gpu-mem-util) GPU_MEM_UTIL="$2"; shift 2;;
    --no-baseline)  RUN_BASELINE=0; shift 1;;   # skip the untrained-base eval
    --runs-dir)     RUNS_DIR="$2"; shift 2;;
    --smoke)        SMOKE=1
                    EXTRA_TRAIN_ARGS+=(--max-steps 2 --max-train-samples 16
                                       --per-device-batch-size 2 --grad-accum 1
                                       --max-length 512)
                    EVAL_LIMIT="5"; shift 1;;
    --train-arg)    EXTRA_TRAIN_ARGS+=("$2"); shift 2;;
    -h|--help)      sed -n '2,30p' "$0"; exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

# ---- fixed config -----------------------------------------------------------
EVAL_TASK="gsm8k"
DATA_FILE="${EXAMPLES_DIR}/gsm8k/${DATA_FILE_NAME}"
TEMPLATES_DIR="${PTB_DIR}/src/eval/templates"
EVAL_SCRIPT="${PTB_DIR}/src/eval/tasks/${EVAL_TASK}/evaluate.py"
MODEL_TAG="$(echo "$MODEL" | tr '/:' '__')"
[[ -z "$RUNS_DIR" ]] && RUNS_DIR="/data/user_data/vijayv/tir5/forger_training_runs_gsm8k_high_lr_lora"
RUN_DIR="${RUNS_DIR}/${VARIANT}/${MODEL_TAG}"
SFT_DIR="${RUN_DIR}/sft"          # the single SFT'd model the eval loads

M_BASE="${RUN_DIR}/metrics_base.json"
M_SFT="${RUN_DIR}/metrics_gsm8k_${VARIANT}.json"
mkdir -p "$RUN_DIR"

[[ -f "$DATA_FILE" ]] || {
  echo "Missing dataset: $DATA_FILE (run examples/gsm8k/convert_to_trl.py)" >&2; exit 1; }

# ---- conda env --------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# L40/L40S and RTX 6000 Ada GPUs have no NVLink/CUDA P2P; NCCL can surface this as a
# misleading "insufficient driver" CUDA error during multi-GPU init. Only disable P2P
# on those cards so we don't slow down NVLink nodes (A100/H100) with working P2P.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
if echo "$GPU_NAME" | grep -qiE 'L40|6000 Ada'; then
  echo "Detected '$GPU_NAME' (no P2P) -> exporting NCCL_P2P_DISABLE=1"
  export NCCL_P2P_DISABLE=1
fi

# ---- GPU count (SLURM-aware, falls back to nvidia-smi) ----------------------
if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  NUM_GPUS="$SLURM_GPUS_ON_NODE"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_GPUS=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
else
  NUM_GPUS=$(nvidia-smi -L | wc -l)
fi

echo "============================================================"
echo " dataset      : gsm8k_${VARIANT}  ->  benchmark: $EVAL_TASK"
echo " model        : $MODEL"
echo " SFT          : single stage (LoRA, lr=$LR x10 in train_sft), epochs=$EPOCHS, no warm-up/chaining"
echo " GPUs (train) : $NUM_GPUS  (FSDP full-shard)"
echo " run dir      : $RUN_DIR"
echo "============================================================"

# ---- helpers ----------------------------------------------------------------
train_sft() {  # <model_in> <data_file> <out_dir> <label> [extra train_sft.py args...]
  local model_in="$1" data="$2" out="$3" label="$4"; shift 4
  echo ">>> Training [$label]: $model_in + $(basename "$data") -> $out"
  # Per-call args ("$@") come before EXTRA_TRAIN_ARGS so --smoke's --max-steps
  # (appended last) still wins; argparse takes the last value for any repeat.
  accelerate launch \
    --config_file "${EXAMPLES_DIR}/training/fsdp.yaml" \
    --num_processes "$NUM_GPUS" \
    "${EXAMPLES_DIR}/training/train_sft.py" \
      --model-id "$model_in" \
      --data-file "$data" \
      --output-dir "$out" \
      --templates-dir "$TEMPLATES_DIR" \
      --learning-rate "$LR" \
      "$@" \
      "${EXTRA_TRAIN_ARGS[@]}"
}

LIMIT_ARG=()
[[ "$EVAL_LIMIT" != "-1" ]] && LIMIT_ARG=(--limit "$EVAL_LIMIT")

run_eval() {  # <model_path> <metrics_file> <label>; fresh vLLM process each call
  local model_path="$1" metrics_file="$2" label="$3"
  echo ">>> Evaluating [$label]: $model_path"
  rm -f "$metrics_file"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python "$EVAL_SCRIPT" \
    --model-path "$model_path" \
    --templates-dir "$TEMPLATES_DIR" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-connections 16 \
    --json-output-file "$metrics_file" \
    "${LIMIT_ARG[@]}" || true
}

# ---- 1) TRAIN (single stage) ------------------------------------------------
# --save-every-epoch: writes intact per-epoch checkpoints (the final model is promoted
# from the last one to dodge the FSDP final-save corruption) and lets you sweep epochs.
echo ">>> [1/1] SFT base on GSM8K ${VARIANT}"
train_sft "$MODEL" "$DATA_FILE" "$SFT_DIR" "base -> gsm8k_${VARIANT}" \
  --epochs "$EPOCHS" --save-every-epoch --use-lora

# ---- 2) EVALUATE ------------------------------------------------------------
# Each eval is a fresh single-GPU vLLM process; the eval forces the per-family chat
# template so base and the SFT'd model are scored with identical prompting.
echo ">>> Evaluating on ${EVAL_TASK} ..."
cd "$RUN_DIR"   # inspect-ai writes ./logs here

if [[ "$RUN_BASELINE" -eq 1 ]]; then
  run_eval "$MODEL" "$M_BASE" "base"
fi
run_eval "$SFT_DIR" "$M_SFT" "base+gsm8k_${VARIANT}"

# ---- results ----------------------------------------------------------------
echo "============================================================"
echo " RESULTS (gsm8k_${VARIANT} / $EVAL_TASK / $MODEL)"
show() {  # pretty-print "<label>: <metrics>" from a metrics json, or note if missing
  local file="$1" label="$2"
  if [[ -f "$file" ]]; then
    printf ' %-14s %s\n' "$label:" "$(python -c "import json,sys;print(json.load(open(sys.argv[1])))" "$file")"
  else
    printf ' %-14s <no metrics -- check eval logs in %s/logs>\n' "$label:" "$RUN_DIR"
  fi
}
if [[ "$RUN_BASELINE" -eq 1 ]]; then
  show "$M_BASE" "base"
fi
show "$M_SFT" "base+gsm8k_${VARIANT}"
echo "============================================================"

# Fail only if the SFT'd model (our actual deliverable) produced no metrics.
[[ -f "$M_SFT" ]] || {
  echo "ERROR: the SFT eval produced no metrics -- see $RUN_DIR/logs" >&2; exit 1; }
