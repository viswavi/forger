#!/usr/bin/env bash
# Train Slime GRPO, then evaluate the final HF checkpoint.
# Usage: sbatch rl/run_rl_training_and_evaluation.sh --rl-data-dir rl/data/full
#        [--init-checkpoint <local-HF-dir>] [--no-thinking] [--dry-run]
#
# Use rl/measure_group_variance.py to check the starting policy: equal-reward
# groups contribute no gradient. The zero-std filter (on by default) drops them
# and changes the training mix; truncated samples are masked from the loss (also
# on by default). See rl/README.md for the recipe and remaining flags.
#
# base_score matches benchmark accuracy; reward also penalizes false
# abstention (-0.1). Optional --doubt-reward <1 grants gameable partial credit
# only for closed traces with --doubt-min matches (default 3).
#
# --no-thinking disables reasoning for rollout and evaluation and reduces
# the default response cap. --fp16 sets both training and rollout dtype;
# use a modest --initial-loss-scale to avoid skipped overflow steps.
#
# --slime-env selects the separate Slime env; --sif selects the container.
# Keep Slime’s Torch stack separate from TRL. --dry-run prints commands
# without conversion or GPU work. --rollout-log-dir stores per-process logs.
#SBATCH --job-name=rl_train_eval_coding
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=90G
#SBATCH --time=24:00:00
#SBATCH --output=rl_train_eval_%j.log

set -euo pipefail
# Validate judge configuration before checkpoint conversion or GPU startup.
JUDGE_MAX_CONSECUTIVE_FAILURES="${JUDGE_MAX_CONSECUTIVE_FAILURES:-5}"
[[ "$JUDGE_MAX_CONSECUTIVE_FAILURES" =~ ^[1-9][0-9]*$ ]] || { echo 'JUDGE_MAX_CONSECUTIVE_FAILURES must be a positive integer' >&2; exit 2; }
export JUDGE_MAX_CONSECUTIVE_FAILURES


# Validate SLURM_SUBMIT_DIR; interactive allocations may inherit a stale path.
CODING_DIR="${SLURM_SUBMIT_DIR:-}"
if [ ! -f "${CODING_DIR}/train_model.sh" ]; then
  CODING_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi
[ -f "${CODING_DIR}/train_model.sh" ] || {
  echo "!!! cannot locate post_data_bench/coding (tried '$CODING_DIR')" >&2; exit 1; }
cd "$CODING_DIR"

# RL runs live apart from ../agent_data_models/, which holds the SFT checkpoints,
# so a run's outputs never mix with the checkpoint it was initialised from. See
# /data/user_data/vijayv/tir5/rl_runs/README.md for the layout.
RL_OUTPUT_ROOT="${RL_OUTPUT_ROOT:-/data/user_data/vijayv/tir5/rl_runs/coding}"
SFT_CHECKPOINT_DIR="/data/user_data/vijayv/tir5/agent_data_models/coding"
QWEN3_4B_LOCAL="/data/user_data/vijayv/tir5/extra_storage/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

SLIME_ENV="${SLIME_ENV:-/home/vijayv/miniconda3/envs/slime}"
SLIME_DEPS="${SLIME_DEPS:-/data/user_data/vijayv/slime_deps}"
SIF=""
RL_DATA_DIR=""
INIT_CHECKPOINT="$QWEN3_4B_LOCAL"
OUTPUT_DIR=""
SPLIT="test"
NO_THINKING=0
# On by default, as in the babel run that took abstention from 7% (SFT) to 51%
# (RL); RL_FILTER_ZERO_STD=0 / RL_DROP_TRUNCATED=0 turn them off (the
# agent-facing rl.agent_training has no flags for them).
FILTER_ZERO_STD="${RL_FILTER_ZERO_STD:-1}"
DROP_TRUNCATED="${RL_DROP_TRUNCATED:-1}"
NUM_ROLLOUT=80
ROLLOUT_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
GLOBAL_BATCH_SIZE=256
MAX_RESPONSE_LEN=""
LR="3e-6"
KL_LOSS_COEF="1e-3"
TEMPERATURE="1.0"
SAVE_INTERVAL=20
EVAL_INTERVAL=10
SKIP_FINAL_EVAL=0
ROLLOUT_LOG_DIR=""
DOUBT_REWARD="0"
DOUBT_MIN="3"
USE_FP16=0
INITIAL_LOSS_SCALE="65536"
DRY_RUN=0
RUN_TAG=""
RUBRIC_REWARDS=0
CUSTOM_RM_PATH="rl.impossible_lcb_rm.custom_rm"
EXTRA_SLIME_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rubric-rewards) RUBRIC_REWARDS=1; shift ;;
    --task) PDB_TASK="$2"; shift 2 ;;
    --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
    --judge-reasoning-effort) JUDGE_REASONING_EFFORT="$2"; shift 2 ;;
    --judge-token-unit-limit) JUDGE_TOKEN_UNIT_LIMIT="$2"; shift 2 ;;
    --judge-token-usage-file) JUDGE_TOKEN_USAGE_FILE="$2"; shift 2 ;;
    --rl-data-dir) RL_DATA_DIR="$2"; shift 2 ;;
    --init-checkpoint) INIT_CHECKPOINT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --sif) SIF="$2"; shift 2 ;;
    --slime-env) SLIME_ENV="$2"; shift 2 ;;
    --slime-deps) SLIME_DEPS="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    --rollout-log-dir) ROLLOUT_LOG_DIR="$2"; shift 2 ;;
    --doubt-reward) DOUBT_REWARD="$2"; shift 2 ;;
    --doubt-min) DOUBT_MIN="$2"; shift 2 ;;
    --fp16) USE_FP16=1; shift ;;
    --initial-loss-scale) INITIAL_LOSS_SCALE="$2"; shift 2 ;;
    --num-rollout) NUM_ROLLOUT="$2"; shift 2 ;;
    --rollout-batch-size) ROLLOUT_BATCH_SIZE="$2"; shift 2 ;;
    --n-samples-per-prompt) N_SAMPLES_PER_PROMPT="$2"; shift 2 ;;
    --global-batch-size) GLOBAL_BATCH_SIZE="$2"; shift 2 ;;
    --max-response-len) MAX_RESPONSE_LEN="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --kl-loss-coef) KL_LOSS_COEF="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
    --eval-interval) EVAL_INTERVAL="$2"; shift 2 ;;
    --no-thinking) NO_THINKING=1; shift ;;
    --filter-zero-std) FILTER_ZERO_STD=1; shift ;;
    --drop-truncated) DROP_TRUNCATED=1; shift ;;
    --skip-final-eval) SKIP_FINAL_EVAL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) EXTRA_SLIME_ARGS+=("$1"); shift ;;
  esac
done

if [ "$RUBRIC_REWARDS" -eq 1 ]; then
  CUSTOM_RM_PATH="rl.rubric_rm.custom_rm"
  # All training knobs remain host-owned. No arbitrary custom RM or eval config
  # can be smuggled into a graded rubric run through the Slime passthrough.
  if [ "${#EXTRA_SLIME_ARGS[@]}" -ne 0 ] || [ "$NO_THINKING" -eq 1 ]; then
    echo "!!! rubric mode requires thinking and disallows extra Slime arguments" >&2
    exit 1
  fi
fi

if [ -z "$RL_DATA_DIR" ]; then
  echo "!!! usage: sbatch $0 --rl-data-dir <dir with train.jsonl/val.jsonl> [--no-thinking] ..." >&2
  exit 1
fi
RL_DATA_DIR="$(cd "$RL_DATA_DIR" && pwd)"
if [ "$RUBRIC_REWARDS" -eq 1 ]; then
  export PDB_TASK="${PDB_TASK:-coding}"
  export JUDGE_MODEL="${JUDGE_MODEL:-litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731}"
  export JUDGE_REASONING_EFFORT="${JUDGE_REASONING_EFFORT:-low}"
  if [ -z "${JUDGE_TOKEN_UNIT_LIMIT:-}" ]; then
    JUDGE_TOKEN_UNIT_LIMIT="$(python3 -c 'import os; from rl.token_budget import default_rl_limit; print(default_rl_limit(os.environ["PDB_TASK"]))')"
  fi
  export JUDGE_TOKEN_UNIT_LIMIT
  export JUDGE_TOKEN_USAGE_FILE="${JUDGE_TOKEN_USAGE_FILE:-$RL_DATA_DIR/../judge_token_usage.json}"
  python3 -c 'from rl.rubric_rm import judge_configuration; judge_configuration()'
fi
for f in train.jsonl val.jsonl; do
  [ -f "$RL_DATA_DIR/$f" ] || { echo "!!! missing $RL_DATA_DIR/$f (run rl/build_rl_data.py)" >&2; exit 1; }
done
if [ -n "$SIF" ]; then
  [ -f "$SIF" ] || { echo "!!! slime image not found: $SIF (run rl/setup_slime_container.sh)" >&2; exit 1; }
else
  [ -x "$SLIME_ENV/bin/python3" ] || { echo "!!! no slime conda env at $SLIME_ENV (run rl/setup_slime_conda.sh)" >&2; exit 1; }
  [ -d "$SLIME_DEPS/Megatron-LM" ] || { echo "!!! no Megatron-LM under $SLIME_DEPS (run rl/setup_slime_conda.sh)" >&2; exit 1; }
fi
[ -d "$INIT_CHECKPOINT" ] || { echo "!!! --init-checkpoint must be a local HF directory: $INIT_CHECKPOINT" >&2; exit 1; }

# Use a shorter cap for no-thinking answers and room for long thinking traces.
if [ -z "$MAX_RESPONSE_LEN" ]; then
  if [ "$NO_THINKING" -eq 1 ]; then MAX_RESPONSE_LEN=3072; else MAX_RESPONSE_LEN=16384; fi
  echo ">>> defaulting --max-response-len -> $MAX_RESPONSE_LEN"
fi

[ -n "$RUN_TAG" ] || RUN_TAG="$(basename "$RL_DATA_DIR")_$(date +%Y%m%d_%H%M%S)"
# A HuggingFace cache path ends in a snapshot hash, which makes an unreadable run
# name; recover the model name from the models--Org--Name component instead.
INIT_TAG="$(basename "$INIT_CHECKPOINT")"
if [[ "$INIT_TAG" =~ ^[0-9a-f]{40}$ ]]; then
  INIT_TAG="$(echo "$INIT_CHECKPOINT" | sed -n 's|.*/models--[^-]*--\([^/]*\)/.*|\1|p')"
  [ -n "$INIT_TAG" ] || INIT_TAG="$(basename "$INIT_CHECKPOINT")"
fi
THINK_TAG=""
[ "$NO_THINKING" -eq 1 ] && THINK_TAG="_no_thinking"
[ -n "$OUTPUT_DIR" ] || OUTPUT_DIR="${RL_OUTPUT_ROOT}/GRPO_${INIT_TAG}_${RUN_TAG}${THINK_TAG}"

# Cache HF-to-Megatron conversion by checkpoint and dtype; bf16/fp16
# conversions are not interchangeable.
PREC_TAG=""
[ "$USE_FP16" -eq 1 ] && PREC_TAG="_fp16"
# Agent-selected checkpoints often share the name 'initial_checkpoint'. Keep
# conversion caches local to each run so another run cannot supply stale weights.
TORCH_DIST="${OUTPUT_DIR}/torch_dist${PREC_TAG}"

CONVERT_PRECISION_ARG=""
[ "$USE_FP16" -eq 1 ] && CONVERT_PRECISION_ARG="--fp16"

# Disable gradient accumulation fusion when Apex’s CUDA extension is absent.
# The converter also needs this because it builds the same GPTModel.
GRAD_FUSION_ARG="--no-gradient-accumulation-fusion"
if [ -z "$SIF" ] && "$SLIME_ENV/bin/python3" -c "import fused_weight_gradient_mlp_cuda" 2>/dev/null; then
  GRAD_FUSION_ARG=""
  echo ">>> apex fused weight-gradient kernel found; leaving gradient accumulation fusion on"
else
  echo ">>> no apex fused weight-gradient kernel; --no-gradient-accumulation-fusion (throughput only)"
fi

NUM_GPUS="$(nvidia-smi -L | wc -l)"
echo ">>> node=$(hostname) gpus=$NUM_GPUS"
echo ">>> init=$INIT_CHECKPOINT"
echo ">>> out=$OUTPUT_DIR"
echo ">>> data=$RL_DATA_DIR (thinking=$([ "$NO_THINKING" -eq 1 ] && echo off || echo on))"

# Run training in the selected Slime env/image and place JIT caches off $HOME.
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-$SLIME_DEPS/.tvm_ffi_cache}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$SLIME_DEPS/.flashinfer}"
mkdir -p "$TVM_FFI_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"

if [ -n "$SIF" ]; then
  SLIME_DIR="/root/slime"
  MEGATRON_DIR="/root/Megatron-LM"
  # Use --cleanenv and a non-login shell so host conda cannot shadow image Python.
  APPTAINER_ARGS=(
    --nv
    --cleanenv
    --bind /data:/data
    --bind /home/vijayv:/home/vijayv
    --env "HF_HOME=${HF_HOME:-/data/user_data/vijayv/tir5/extra_storage/hf_cache}"
    --env "HF_HUB_OFFLINE=1"
    --env "PYTHONUNBUFFERED=1"
    --env "IMPOSSIBLE_LCB_DIR=$CODING_DIR"
  )
  # --cleanenv means exported host variables do not reach the container, so the
  # verifier's knobs are forwarded explicitly, read at call time (they are set
  # further down, after this function is defined).
  run_in_slime() {
    APPTAINERENV_LITELLM_JUDGE_API_KEY="${LITELLM_JUDGE_API_KEY:-}" \
    apptainer exec "${APPTAINER_ARGS[@]}" \
      --env "LITELLM_JUDGE_BASE_URL=${LITELLM_JUDGE_BASE_URL:-}" \
      --env "JUDGE_MAX_CONSECUTIVE_FAILURES=${JUDGE_MAX_CONSECUTIVE_FAILURES:-5}" \
      --env "LITELLM_BASE_URL=${LITELLM_BASE_URL:-}" \
      --env "JUDGE_QUOTA_STOP_FILE=${JUDGE_QUOTA_STOP_FILE:-}" \
      --env "JUDGE_MODEL=${JUDGE_MODEL:-}" \
      --env "JUDGE_REASONING_EFFORT=${JUDGE_REASONING_EFFORT:-}" \
      --env "JUDGE_TOKEN_UNIT_LIMIT=${JUDGE_TOKEN_UNIT_LIMIT:-}" \
      --env "JUDGE_TOKEN_USAGE_FILE=${JUDGE_TOKEN_USAGE_FILE:-}" \
      --env "PDB_TASK=${PDB_TASK:-coding}" \
      --env "RUBRIC_REWARDS=$RUBRIC_REWARDS" \
      --env "LCB_RM_WORKERS=${LCB_RM_WORKERS:-16}" \
      --env "LCB_RM_DUMP=${LCB_RM_DUMP:-}" \
      --env "LCB_RM_LOG_DIR=${LCB_RM_LOG_DIR:-}" \
      --env "LCB_RM_DOUBT_REWARD=${LCB_RM_DOUBT_REWARD:-0}" \
      --env "LCB_RM_DOUBT_MIN=${LCB_RM_DOUBT_MIN:-3}" \
      --env "TVM_FFI_CACHE_DIR=$TVM_FFI_CACHE_DIR" \
      --env "FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE" \
      "$SIF" bash -c "$1"
  }
  echo ">>> slime runtime: apptainer image $SIF"
else
  SLIME_DIR="${SLIME_SRC:-/home/vijayv/slime}"
  [[ "$SLIME_ENV" != /usr ]] || SLIME_DIR="${SLIME_SRC:-/opt/slime}"
  MEGATRON_DIR="$SLIME_DEPS/Megatron-LM"
  export HF_HOME="${HF_HOME:-/data/user_data/vijayv/tir5/extra_storage/hf_cache}"
  export HF_HUB_OFFLINE=1
  export PYTHONUNBUFFERED=1
  export IMPOSSIBLE_LCB_DIR="$CODING_DIR"
  # Activate in a subshell to stream output without changing the caller’s env.
  # Use the env’s CUDA toolkit rather than the host nvcc.
  CONDA_BASE=""
  [[ "$SLIME_ENV" == /usr ]] || CONDA_BASE="$(conda info --base)"
  # Prefer the env’s libstdc++ for JIT kernels requiring newer GLIBCXX symbols.
  # Keep JIT caches off $HOME.
  run_in_slime() {
    if [[ "$SLIME_ENV" == /usr ]]; then
      # The combined image's system stack; TRL must not shadow its entry points.
      PATH=/usr/local/bin:/usr/bin:/bin bash -c "set -euo pipefail; $1"
    else
      bash -c "
        set -euo pipefail
        source '$CONDA_BASE/etc/profile.d/conda.sh'
        conda activate '$SLIME_ENV'
        export CUDA_HOME=\"\$CONDA_PREFIX\"
        export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\"
        hash -r
        $1"
    fi
  }
  echo ">>> slime runtime: conda env $SLIME_ENV"
  if [[ "$SLIME_ENV" == /usr && "$DRY_RUN" -eq 0 ]] && ! /usr/bin/python3 -c "import RestrictedPython" 2>/dev/null; then
    # The combined image has RestrictedPython only in the TRL env, but rubric
    # rewards import it in Slime's Python (every agent run had to pip-install it
    # itself). It is pure Python and both are 3.12; the container overlay is
    # writable.
    TRL_SITE=/home/vijayv/miniconda3/envs/trl/lib/python3.12/site-packages
    cp -r "$TRL_SITE/RestrictedPython" /usr/local/lib/python3.12/dist-packages/
    /usr/bin/python3 -c "import RestrictedPython"
    echo ">>> copied RestrictedPython from the TRL env into Slime's Python"
  fi
fi

# ---------------------------------------------------------------------------
# Step 1: HF -> Megatron torch_dist (cached).
# ---------------------------------------------------------------------------
if [ -f "$TORCH_DIST/latest_checkpointed_iteration.txt" ]; then
  echo ">>> reusing torch_dist checkpoint at $TORCH_DIST"
elif [ "$DRY_RUN" -eq 1 ]; then
  echo ">>> --dry-run: would convert $INIT_CHECKPOINT -> $TORCH_DIST"
else
  echo ">>> converting $INIT_CHECKPOINT -> $TORCH_DIST"
  mkdir -p "$TORCH_DIST"
  run_in_slime "
    set -euo pipefail
    cd $SLIME_DIR
    source scripts/models/qwen3-4B.sh
    PYTHONPATH=$MEGATRON_DIR python3 tools/convert_hf_to_torch_dist.py \
      \"\${MODEL_ARGS[@]}\" \
      $CONVERT_PRECISION_ARG $GRAD_FUSION_ARG \
      --hf-checkpoint '$INIT_CHECKPOINT' \
      --save '$TORCH_DIST'
  "
fi

# ---------------------------------------------------------------------------
# Step 2: GRPO.
# ---------------------------------------------------------------------------
CHAT_TEMPLATE_KWARGS='{"enable_thinking": true}'
[ "$NO_THINKING" -eq 1 ] && CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'

# Throughput on 8x H100 80GB with a 4B model (Orchard 2026-10-26: the old
# TP=2 / 8192 tokens / 0.5 settings, meant for 48GB cards, used ~45 of 80 GB).
# Megatron and SGLang both run at TP_SIZE; Megatron disables sequence
# parallelism by itself at TP=1. None of these change the training math.
TP_SIZE="${RL_TP_SIZE:-1}"
MAX_TOKENS_PER_GPU="${RL_MAX_TOKENS_PER_GPU:-24576}"
SGLANG_MEM_FRACTION="${RL_SGLANG_MEM_FRACTION:-0.7}"
echo ">>> throughput: TP=$TP_SIZE, max tokens/GPU $MAX_TOKENS_PER_GPU, SGLang memory fraction $SGLANG_MEM_FRACTION"

DYNAMIC_FILTER_ARGS=""
if [ "$FILTER_ZERO_STD" -eq 1 ]; then
  # Sample RL_OVER_SAMPLING_FACTOR x the batch per wave: most groups come back
  # all-same and get dropped (Orchard 162166 step 4: 81+ dropped for 32 kept),
  # so one-batch waves meant several refills per step.
  OVER_SAMPLING=$(( ROLLOUT_BATCH_SIZE * ${RL_OVER_SAMPLING_FACTOR:-2} ))
  DYNAMIC_FILTER_ARGS="--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std --over-sampling-batch-size $OVER_SAMPLING"
  echo ">>> dropping zero-variance prompt groups (this reshapes the training mix; see header); sampling $OVER_SAMPLING prompts per wave"
fi
SAMPLE_FILTER_ARGS=""
if [ "$DROP_TRUNCATED" -eq 1 ]; then
  SAMPLE_FILTER_ARGS="--rollout-sample-filter-path rl.sample_filters.drop_truncated"
  echo ">>> masking truncated samples from the loss (rl.sample_filters.drop_truncated)"
fi

# The verifier grades in a spawn-context process pool sized here. It executes
# untrusted model code, so it wants real cores, not GPU time.
PRECISION_ARGS=""
if [ "$USE_FP16" -eq 1 ]; then
  PRECISION_ARGS="--fp16 --initial-loss-scale $INITIAL_LOSS_SCALE --sglang-dtype float16"
  echo ">>> fp16: training and rollout both in fp16, initial loss scale $INITIAL_LOSS_SCALE"
else
  echo ">>> bf16 (default); pass --fp16 to train and roll out in fp16 instead"
fi

RM_WORKERS="${LCB_RM_WORKERS:-$(( $(nproc) / 2 ))}"
[ "$RM_WORKERS" -ge 4 ] || RM_WORKERS=4

# Store per-process rollout/reward logs beside checkpoints; summarize_rollouts
# reads them for analysis.
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-$OUTPUT_DIR/logs/rollouts}"
# Only a real run creates directories -- a --dry-run that leaves empty run dirs
# behind litters the output root with plausible-looking runs that never happened.
[ "$DRY_RUN" -eq 1 ] || mkdir -p "$OUTPUT_DIR/ckpt" "$OUTPUT_DIR/logs" "$ROLLOUT_LOG_DIR"
REWARD_DUMP="$OUTPUT_DIR/logs/reward_dump.jsonl"
echo ">>> rollout log: $ROLLOUT_LOG_DIR"
if [ "$DOUBT_REWARD" != "0" ]; then
  if [ "$NO_THINKING" -eq 1 ]; then
    echo "!!! --doubt-reward needs a thinking trace to score; it never fires with --no-thinking" >&2
    exit 1
  fi
  echo ">>> process reward: $DOUBT_REWARD for an impossible row whose closed trace has >= $DOUBT_MIN doubt matches"
fi

TRAIN_SCRIPT="
set -euo pipefail
cd $SLIME_DIR
source scripts/models/qwen3-4B.sh

# rl/ lives in the study's repo, not in the image, so both the driver and every
# ray worker need it importable for --custom-rm-path to resolve.
export PYTHONPATH='$CODING_DIR:$MEGATRON_DIR'
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Single-node Ray; train.py reaches the GCS on loopback. Orchard presets
# MASTER_ADDR to the node IP. Ray still registers the node under its own IP.
# RL_RAY_NODE_IP overrides the address.
export MASTER_ADDR=\${RL_RAY_NODE_IP:-127.0.0.1}
# By default the raylet holds the first driver's registration until it has
# prestarted one worker per CPU (208 on an Orchard node). Inside the fakeroot
# container those workers sometimes never register, so train.py waited forever
# with idle GPUs (162182, 162194, 162222). Start workers on demand instead.
export RAY_prestart_worker_first_driver=0

# Use a short per-user/job Ray directory to avoid shared /tmp ownership
# and cleanup races while staying within Unix socket path limits.
RAY_TEMP_DIR=/tmp/ray-\$(id -un)-\${SLURM_JOB_ID:-\$\$}

ray stop --force 2>/dev/null || true
ray start --head --node-ip-address \$MASTER_ADDR --num-gpus $NUM_GPUS \
  --temp-dir \$RAY_TEMP_DIR \
  --disable-usage-stats --dashboard-host=127.0.0.1 --dashboard-port=8265

RUNTIME_ENV_JSON='{\"env_vars\": {
  \"PYTHONPATH\": \"$CODING_DIR:$MEGATRON_DIR\",
  \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
  \"IMPOSSIBLE_LCB_DIR\": \"$CODING_DIR\",
  \"LCB_RM_WORKERS\": \"$RM_WORKERS\",
  \"LCB_RM_DUMP\": \"$REWARD_DUMP\",
  \"LCB_RM_LOG_DIR\": \"$ROLLOUT_LOG_DIR\",
  \"LCB_RM_DOUBT_REWARD\": \"$DOUBT_REWARD\",
  \"LCB_RM_DOUBT_MIN\": \"$DOUBT_MIN\",
  \"TVM_FFI_CACHE_DIR\": \"$TVM_FFI_CACHE_DIR\",
  \"FLASHINFER_WORKSPACE_BASE\": \"$FLASHINFER_WORKSPACE_BASE\",
  \"HF_HUB_OFFLINE\": \"1\"
}}'

# Write credentials to a mode-600 runtime-env file rather than shell tracing
# them or putting them in ray's command-line arguments.
RUNTIME_ENV_FILE=\$(mktemp /dev/shm/rubric-ray-env.\$\$.XXXXXX.json)
trap 'rm -f \"\$RUNTIME_ENV_FILE\"; ray stop --force >/dev/null 2>&1 || true' EXIT
python3 '$CODING_DIR/rl/rubric_runtime_env.py' \"\$RUNTIME_ENV_JSON\" \"\$RUNTIME_ENV_FILE\"
RAY_JOB_ID=\$(python3 -c 'import uuid; print(uuid.uuid4().hex)')
TRAIN_ARGS=(
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node $NUM_GPUS \
  --colocate \
  \"\${MODEL_ARGS[@]}\" \
  --hf-checkpoint '$INIT_CHECKPOINT' \
  --ref-load '$TORCH_DIST' \
  --load '$OUTPUT_DIR/ckpt' \
  --save '$OUTPUT_DIR/ckpt' \
  --save-hf '$OUTPUT_DIR/hf_rollout{rollout_id}' \
  --save-interval $SAVE_INTERVAL \
  --prompt-data '$RL_DATA_DIR/train.jsonl' \
  --input-key prompt \
  --label-key label \
  --metadata-key metadata \
  --apply-chat-template \
  --apply-chat-template-kwargs '$CHAT_TEMPLATE_KWARGS' \
  --rollout-shuffle \
  --custom-rm-path $CUSTOM_RM_PATH \
  --num-rollout $NUM_ROLLOUT \
  --rollout-batch-size $ROLLOUT_BATCH_SIZE \
  --n-samples-per-prompt $N_SAMPLES_PER_PROMPT \
  --rollout-max-response-len $MAX_RESPONSE_LEN \
  --rollout-temperature $TEMPERATURE \
  --global-batch-size $GLOBAL_BATCH_SIZE \
  --balance-data \
  $DYNAMIC_FILTER_ARGS \
  $SAMPLE_FILTER_ARGS \
  --eval-interval $EVAL_INTERVAL \
  --eval-prompt-data generated_val '$RL_DATA_DIR/val.jsonl' \
  --n-samples-per-eval-prompt 4 \
  --eval-max-response-len $MAX_RESPONSE_LEN \
  --advantage-estimator grpo \
  --use-kl-loss \
  --kl-loss-coef $KL_LOSS_COEF \
  --kl-loss-type low_var_kl \
  --entropy-coef 0.00 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --optimizer adam \
  --lr $LR \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --tensor-model-parallel-size $TP_SIZE \
  --sequence-parallel \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu $MAX_TOKENS_PER_GPU \
  --rollout-num-gpus-per-engine $TP_SIZE \
  --sglang-mem-fraction-static $SGLANG_MEM_FRACTION \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  $PRECISION_ARGS \
  $GRAD_FUSION_ARG \
  ${EXTRA_SLIME_ARGS[*]:-}
)

if [ \"\${RL_RAY_JOB_SUBMIT:-0}\" = 1 ]; then
  # Ray's job API. ray start returns before the dashboard's job agent is up, and
  # submitting then can fail with HTTP 504 (Orchard 162111, 162165): retry
  # unless the job did get through.
  submitted=0
  for attempt in 1 2 3 4 5 6; do
    if ray job submit --no-wait --address=http://127.0.0.1:8265 --submission-id=\"\$RAY_JOB_ID\" \
        --runtime-env=\"\$RUNTIME_ENV_FILE\" -- python3 train.py \"\${TRAIN_ARGS[@]}\"; then
      submitted=1; break
    fi
    if ray job status --address=http://127.0.0.1:8265 \"\$RAY_JOB_ID\" >/dev/null 2>&1; then submitted=1; break; fi
    echo \">>> ray job submit failed (attempt \$attempt); retrying in 20s\"
    sleep 20
  done
  [ \"\$submitted\" = 1 ] || { echo '!!! could not submit the RL job to ray' >&2; exit 1; }
  ray job logs --address=http://127.0.0.1:8265 --follow \"\$RAY_JOB_ID\" || true
  python3 '$CODING_DIR/rl/ray_job_status.py' \"\$RAY_JOB_ID\"
else
  # Default: run train.py as the driver on the cluster ray start launched. The
  # dashboard's job agent can stay unresponsive (Orchard 162173: every
  # submission waited 300 s, then HTTP 504). Workers inherit this shell's
  # exports (PYTHONPATH, reward and judge settings) through ray start.
  echo \">>> running train.py as the ray driver (RL_RAY_JOB_SUBMIT=1 uses ray job submit)\"
  RAY_ADDRESS=\"\$MASTER_ADDR:6379\" python3 train.py \"\${TRAIN_ARGS[@]}\"
fi
ray stop --force 2>/dev/null || true
"

if [ "$DRY_RUN" -eq 1 ]; then
  echo "----- would run in ${SIF:-$SLIME_ENV} -----"
  echo "$TRAIN_SCRIPT"
  echo "----- (--dry-run: stopping here) -----"
  exit 0
fi

export RUBRIC_REWARDS
export LCB_RM_WORKERS="$RM_WORKERS"
export LCB_RM_DUMP="$REWARD_DUMP"
export LCB_RM_LOG_DIR="$ROLLOUT_LOG_DIR"
export LCB_RM_DOUBT_REWARD="$DOUBT_REWARD"
export LCB_RM_DOUBT_MIN="$DOUBT_MIN"
# A unique marker directory per attempt prevents stale quota stops on resume.
JUDGE_STOP_DIR="$(mktemp -d "$OUTPUT_DIR/logs/judge-stop.XXXXXX")"
export JUDGE_QUOTA_STOP_FILE="$JUDGE_STOP_DIR/quota.json"
set +e
run_in_slime "$TRAIN_SCRIPT"
TRAIN_EXIT=$?
set -e

# Only a confirmed quota stop is recoverable; unrelated training failures remain
# errors. Ray cleanup has completed before inspecting saved checkpoint files.
FINAL_HF="$(python3 "$CODING_DIR/rl/rl_result.py" \
  --output-dir "$OUTPUT_DIR" --init-checkpoint "$INIT_CHECKPOINT" \
  --training-exit-code "$TRAIN_EXIT" --quota-file "$JUDGE_QUOTA_STOP_FILE")"
echo "FINAL_CHECKPOINT=$FINAL_HF"

# Evaluate the last HF checkpoint with Slime’s SGLang serving backend.
if [ "$SKIP_FINAL_EVAL" -eq 1 ]; then
  echo ">>> --skip-final-eval: stopping after training. HF checkpoints in $OUTPUT_DIR"
  exit 0
fi

echo ">>> evaluating $FINAL_HF"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SLIME_ENV"
cd "$CODING_DIR"

EVAL_ARGS=()
[ "$NO_THINKING" -eq 1 ] && EVAL_ARGS+=(--no-thinking)
./evaluate_model.sh --server-backend sglang --split "$SPLIT" --model-path "$FINAL_HF" "${EVAL_ARGS[@]}"
