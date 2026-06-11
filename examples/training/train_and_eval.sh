#!/bin/bash
# =============================================================================
# train_and_eval.sh -- SFT a base model on a Forger dataset (optionally on top of
# a LIMA-finetuned checkpoint) and evaluate on the matching PostTrainBench
# benchmark. A SLURM user launches this and finds 4 models trained + scored.
#
#   sbatch train_and_eval.sh                          # defaults: basic_math + Qwen3-1.7B-Base
#   bash   train_and_eval.sh --dataset coding_questions
#   bash   train_and_eval.sh --dataset basic_math --model Qwen/Qwen3-4B-Base
#
# Dataset -> benchmark mapping:
#   basic_math       -> GSM8K     (gsm8k/evaluate.py,     10-shot, exact match)
#   coding_questions -> HumanEval (humaneval/evaluate.py, pass@1)
#
# The 4 evaluated models:
#   base             the untrained base model
#   base+LIMA        base SFT'd on LIMA (general instruction data)
#   base+task        base SFT'd on the task dataset
#   base+LIMA+task   the LIMA checkpoint SFT'd on the task dataset
#
# The LIMA checkpoint is model-specific (not task-specific), so it is cached at
# $CKPT_DIR/<model>_lima and reused across datasets: it is trained only if absent.
#
# Multi-GPU is automatic: all visible GPUs are used for FSDP training. Works on
# the current 2-GPU node and on the 4- or 8-GPU nodes we'll use for real runs.
# =============================================================================
#SBATCH --job-name=sft_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=train_and_eval_%j.log

set -euo pipefail

# ---- defaults ---------------------------------------------------------------
DATASET="basic_math"
MODEL="Qwen/Qwen3-1.7B-Base"
CONDA_ENV="trl"
EXAMPLES_DIR="/home/vijayv/forger/examples"
PTB_DIR="/home/vijayv/PostTrainBench"
CKPT_DIR="/data/user_data/vijayv/tir5/checkpoints"  # shared LIMA checkpoints live here
RUNS_DIR=""
# Lower-LR / more-epochs recipe: SFT brittleness to the eval prompt grows with training
# DOSE, so we train gentler -- 5x lower LR with 5x the epochs -- to reach a similar fit
# while staying closer to the (robust) base model.
LR="2e-6"              # 5x below the train_sft.py 1e-5 default
EPOCHS="15"            # task-data SFT epochs (5x)
LIMA_EPOCHS="75"       # LIMA SFT epochs (5x the paper's 15, at 5x lower LR)
LIMA_WEIGHT_DECAY="0.1"  # LIMA paper value; regularizes the long run
EVAL_LIMIT="-1"        # -1 = full benchmark; set small for quick checks
GPU_MEM_UTIL="0.6"     # vLLM gpu_memory_utilization for the eval phase
RUN_BASELINE=1         # also eval the two non-task models (base, base+LIMA)
SMOKE=0                # cheap end-to-end check
EXTRA_TRAIN_ARGS=()

# ---- arg parsing ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)        DATASET="$2"; shift 2;;
    --model)          MODEL="$2"; shift 2;;
    --epochs)         EPOCHS="$2"; shift 2;;
    --lima-epochs)    LIMA_EPOCHS="$2"; shift 2;;
    --lr)             LR="$2"; shift 2;;
    --eval-limit)     EVAL_LIMIT="$2"; shift 2;;
    --gpu-mem-util)   GPU_MEM_UTIL="$2"; shift 2;;
    --ckpt-dir)       CKPT_DIR="$2"; shift 2;;
    --no-baseline)    RUN_BASELINE=0; shift 1;;   # skip the base & base+LIMA evals
    --runs-dir)       RUNS_DIR="$2"; shift 2;;
    --smoke)          SMOKE=1
                      EXTRA_TRAIN_ARGS+=(--max-steps 2 --max-train-samples 16
                                         --per-device-batch-size 2 --grad-accum 1
                                         --max-length 512)
                      EVAL_LIMIT="5"; shift 1;;
    --train-arg)      EXTRA_TRAIN_ARGS+=("$2"); shift 2;;
    -h|--help)        sed -n '2,30p' "$0"; exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

# ---- dataset -> data file + eval task --------------------------------------
case "$DATASET" in
  basic_math)
    DATA_FILE="${EXAMPLES_DIR}/basic_math/basic_math_trl_format.jsonl"
    EVAL_TASK="gsm8k";;
  coding_questions)
    DATA_FILE="${EXAMPLES_DIR}/coding_questions/coding_questions_trl_format.jsonl"
    EVAL_TASK="humaneval";;
  *)
    echo "Unsupported --dataset '$DATASET' (use basic_math or coding_questions)" >&2
    exit 1;;
esac

TEMPLATES_DIR="${PTB_DIR}/src/eval/templates"
EVAL_SCRIPT="${PTB_DIR}/src/eval/tasks/${EVAL_TASK}/evaluate.py"
LIMA_DATA="${EXAMPLES_DIR}/LIMA/lima_trl.jsonl"
MODEL_TAG="$(echo "$MODEL" | tr '/:' '__')"
MODEL_BASENAME="$(basename "$MODEL")"
[[ -z "$RUNS_DIR" ]] && RUNS_DIR="${EXAMPLES_DIR}/training/runs"
RUN_DIR="${RUNS_DIR}/${DATASET}/${MODEL_TAG}"

# Output model dirs (the two task-trained models).
TASK_BASE_DIR="${RUN_DIR}/base_task"        # base + task
TASK_LIMA_DIR="${RUN_DIR}/lima_task"        # base + LIMA + task

# Shared, cached LIMA checkpoint (per base model). In smoke mode we use a local
# throwaway path so a 2-step smoke run never poisons the real shared cache.
if [[ "$SMOKE" -eq 1 ]]; then
  LIMA_CKPT="${RUN_DIR}/_lima_smoke"
else
  LIMA_CKPT="${CKPT_DIR}/${MODEL_BASENAME}_lima"
fi

# Metrics (one per evaluated model), all under the run dir.
M_BASE="${RUN_DIR}/metrics_base.json"
M_LIMA="${RUN_DIR}/metrics_lima.json"
M_TASK="${RUN_DIR}/metrics_base_task.json"
M_LIMA_TASK="${RUN_DIR}/metrics_lima_task.json"
mkdir -p "$RUN_DIR" "$(dirname "$LIMA_CKPT")"

# ---- conda env --------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# L40/L40S and RTX 6000 Ada GPUs have no NVLink/CUDA P2P; NCCL can surface this as
# a misleading "insufficient driver" CUDA error during multi-GPU init. Only disable
# P2P on those cards so we don't slow down NVLink nodes (A100/H100) that have working
# P2P. nvidia-smi reports the 6000 Ada as "NVIDIA RTX 6000 Ada Generation".
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
echo " dataset      : $DATASET  ->  benchmark: $EVAL_TASK"
echo " model        : $MODEL"
echo " epochs       : LIMA=$LIMA_EPOCHS (wd=$LIMA_WEIGHT_DECAY), task=$EPOCHS, lr=$LR"
echo " GPUs (train) : $NUM_GPUS  (FSDP full-shard)"
echo " run dir      : $RUN_DIR"
echo " LIMA ckpt    : $LIMA_CKPT"
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

# ---- 1) TRAIN ---------------------------------------------------------------
# 1a) LIMA stage (cached): train base on LIMA only if the checkpoint is missing.
if [[ -f "$LIMA_CKPT/config.json" ]]; then
  echo ">>> [1/3] LIMA checkpoint exists, reusing: $LIMA_CKPT"
else
  echo ">>> [1/3] LIMA checkpoint missing -> training base on LIMA"
  # --save-every-epoch + --resume make the long warm-up restartable: if the job is
  # cancelled, just relaunch (same paths, any job name) and it continues from the
  # latest checkpoint toward the same --epochs total.
  train_sft "$MODEL" "$LIMA_DATA" "$LIMA_CKPT" "base -> LIMA" \
    --epochs "$LIMA_EPOCHS" --weight-decay "$LIMA_WEIGHT_DECAY" --save-every-epoch --resume
fi

# 1b) task SFT from the base model            -> base+task
echo ">>> [2/3] Task SFT from base"
train_sft "$MODEL" "$DATA_FILE" "$TASK_BASE_DIR" "base -> task" --epochs "$EPOCHS" --save-every-epoch

# 1c) task SFT from the LIMA checkpoint        -> base+LIMA+task
echo ">>> [3/3] Task SFT from LIMA checkpoint"
train_sft "$LIMA_CKPT" "$DATA_FILE" "$TASK_LIMA_DIR" "base+LIMA -> task" --epochs "$EPOCHS" --save-every-epoch

# ---- 2) EVALUATE 4 models ---------------------------------------------------
# Each eval is a fresh single-GPU vLLM process, run sequentially (no GPU-memory
# contention). The eval forces the per-family chat template, so all 4 models are
# scored with identical prompting.
echo ">>> Evaluating on ${EVAL_TASK} ..."
cd "$RUN_DIR"   # inspect-ai writes ./logs here

if [[ "$RUN_BASELINE" -eq 1 ]]; then
  run_eval "$MODEL"     "$M_BASE" "base"
  run_eval "$LIMA_CKPT" "$M_LIMA" "base+LIMA"
fi
run_eval "$TASK_BASE_DIR" "$M_TASK"      "base+task"
run_eval "$TASK_LIMA_DIR" "$M_LIMA_TASK" "base+LIMA+task"

# ---- results ----------------------------------------------------------------
echo "============================================================"
echo " RESULTS ($DATASET / $EVAL_TASK / $MODEL)"
show() {  # pretty-print "<label>: <metrics>" from a metrics json, or note if missing
  local file="$1" label="$2"
  if [[ -f "$file" ]]; then
    printf ' %-16s %s\n' "$label:" "$(python -c "import json,sys;print(json.load(open(sys.argv[1])))" "$file")"
  else
    printf ' %-16s <no metrics -- check eval logs in %s/logs>\n' "$label:" "$RUN_DIR"
  fi
}
if [[ "$RUN_BASELINE" -eq 1 ]]; then
  show "$M_BASE" "base"
  show "$M_LIMA" "base+LIMA"
fi
show "$M_TASK"      "base+task"
show "$M_LIMA_TASK" "base+LIMA+task"
echo "============================================================"

# Fail only if a task-trained model (our actual deliverable) produced no metrics.
[[ -f "$M_TASK" && -f "$M_LIMA_TASK" ]] || {
  echo "ERROR: a task-trained eval produced no metrics -- see $RUN_DIR/logs" >&2; exit 1; }
