#!/bin/bash
# SFT generated coding data; evaluate separately with evaluate_model.sh.
# LoRA is the default: train_sft.py scales LR 10x and merges adapters.
# --full-finetune uses the raw LR and needs more GPU memory.
#SBATCH --job-name=sft_coding
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=90G
#SBATCH --time=12:00:00
#SBATCH --output=train_model_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$SCRIPT_DIR/prompt_building.py" ]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR:?Submit from the task source directory}"
fi
[ -f "$SCRIPT_DIR/prompt_building.py" ] || { echo "Task source directory not found" >&2; exit 1; }


# ---- defaults ---------------------------------------------------------------
MODEL="Qwen/Qwen3-4B"
CONDA_ENV="${SFT_CONDA_ENV:-slime}"
EXAMPLES_DIR="${EXAMPLES_DIR:-$(cd "$SCRIPT_DIR/../../examples" && pwd)}"
DEFAULT_OUTPUT_BASE="/data/user_data/vijayv/tir5/agent_data_models/coding"
DATA_FILE=""
DATA_FILE_NAME=""
OUTPUT_DIR=""
LR="1e-6"
EPOCHS="4"
# Match convert_to_trl.py’s length filter to avoid truncated targets.
MAX_LENGTH="18800"
FULL_FINETUNE=0
EXTRA_TRAIN_ARGS=()

# ---- arg parsing ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda-env)     CONDA_ENV="$2"; shift 2;;
    --model)          MODEL="$2"; shift 2;;
    --data-file)      DATA_FILE="$2"; shift 2;;
    --data-file-name) DATA_FILE_NAME="$2"; shift 2;;   # resolved relative to this directory
    --output-dir)     OUTPUT_DIR="$2"; shift 2;;
    --epochs)         EPOCHS="$2"; shift 2;;
    --lr)             LR="$2"; shift 2;;
    --max-length)     MAX_LENGTH="$2"; shift 2;;
    --full-finetune)  FULL_FINETUNE=1; shift 1;;
    --smoke)          EXTRA_TRAIN_ARGS+=(--max-steps 2 --max-train-samples 16
                                         --per-device-batch-size 2 --grad-accum 1
                                         --max-length 512); shift 1;;
    --train-arg)      EXTRA_TRAIN_ARGS+=("$2"); shift 2;;
    -h|--help)        sed -n '2,12p' "$0"; exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

# ---- fixed config -----------------------------------------------------------
if [[ -n "$DATA_FILE" && -n "$DATA_FILE_NAME" ]]; then
  echo "Pass only one of --data-file / --data-file-name" >&2; exit 1
fi
[[ -z "$DATA_FILE" && -n "$DATA_FILE_NAME" ]] && DATA_FILE="${SCRIPT_DIR}/${DATA_FILE_NAME}"
[[ -z "$DATA_FILE" ]] && { echo "Must pass --data-file <path to generated training data>" >&2; exit 1; }
[[ -f "$DATA_FILE" ]] || { echo "Missing dataset: $DATA_FILE" >&2; exit 1; }

MODEL_TAG="$(echo "$MODEL" | tr '/:' '__')"
[[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="${DEFAULT_OUTPUT_BASE}/${MODEL_TAG}_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
SFT_DIR="$OUTPUT_DIR"
mkdir -p "$(dirname "$SFT_DIR")"

# ---- conda env --------------------------------------------------------------
if [[ -x "$CONDA_ENV/bin/python" ]]; then
  export PATH="$CONDA_ENV/bin:$PATH"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

# Expandable segments reduce fragmentation from variable-length loss logits.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# L40/L40S and RTX 6000 Ada GPUs have no NVLink/CUDA P2P; NCCL can surface this
# as a misleading "insufficient driver" CUDA error during multi-GPU init.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
if echo "$GPU_NAME" | grep -qiE 'L40|6000 Ada'; then
  echo "Detected '$GPU_NAME' (no P2P) -> exporting NCCL_P2P_DISABLE=1"
  export NCCL_P2P_DISABLE=1
fi

# ---- GPU count (SLURM-aware, falls back to nvidia-smi) ----------------------
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_GPUS=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
else
  NUM_GPUS=$(nvidia-smi -L | wc -l)
fi

GLOBAL_BATCH=${SFT_GLOBAL_BATCH:-128}
[ "$NUM_GPUS" -gt 0 ] && [ "$((GLOBAL_BATCH % NUM_GPUS))" -eq 0 ] || { echo "Global batch must divide GPU count" >&2; exit 1; }
GRAD_ACCUM=$((GLOBAL_BATCH / NUM_GPUS))
echo "============================================================"
echo " model        : $MODEL"
echo " data         : $DATA_FILE"
echo " GPUs (train) : $NUM_GPUS  (FSDP full-shard)"
echo " output dir   : $SFT_DIR"
if [[ "$FULL_FINETUNE" -eq 1 ]]; then
  echo " tuning       : full-parameter (--lr used as-is, no 10x scaling)"
else
  echo " tuning       : LoRA (--lr auto-scaled 10x by train_sft.py)"
fi
echo "============================================================"

# Download once before launching ranks to avoid shared HF-cache races.
if [[ "$MODEL" != /* && ! -d "$MODEL" ]]; then
  echo ">>> pre-downloading $MODEL to the local HF cache (single process, avoids a multi-rank download race)"
  python -c 'from huggingface_hub import snapshot_download; import sys; snapshot_download(sys.argv[1])' "$MODEL"
fi

# Use the completed cache offline so ranks cannot race on Hub revalidation.
export HF_HUB_OFFLINE=1

# Save in-loop epoch checkpoints for intact FSDP output and use the default
# chat template to match evaluation. Derive a per-job rendezvous port to
# avoid collisions between jobs sharing a node.
MAIN_PROCESS_PORT=$(( 20000 + (${SLURM_JOB_ID:-$RANDOM} % 40000) ))
# LoRA on every linear layer: attention-only LoRA underperforms MLP LoRA
# (https://thinkingmachines.ai/blog/lora/). alpha=32 with the 1/r scaling keeps
# the best LR roughly rank-independent.
LORA_ARGS=()
[[ "$FULL_FINETUNE" -eq 0 ]] && LORA_ARGS=(--use-lora --lora-r 256 --lora-alpha 32
  --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj)
accelerate launch \
  --config_file "${EXAMPLES_DIR}/training/fsdp.yaml" \
  --num_processes "$NUM_GPUS" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  "${EXAMPLES_DIR}/training/train_sft.py" \
    --model-id "$MODEL" \
    --data-file "$DATA_FILE" \
    --output-dir "$SFT_DIR" \
    --learning-rate "$LR" \
    --epochs "$EPOCHS" --save-every-epoch --max-length "$MAX_LENGTH" \
    "${LORA_ARGS[@]}" --use-default-chat-template --assistant-only-loss \
    --per-device-batch-size 1 --grad-accum "$GRAD_ACCUM" \
    "${EXTRA_TRAIN_ARGS[@]}"

echo "MODEL_OUTPUT_DIR=${SFT_DIR}"
echo ">>> Model is trained and saved at ${SFT_DIR}: $(ls "$SFT_DIR")"
