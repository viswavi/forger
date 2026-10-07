#!/usr/bin/env bash
# Sweep harmonized data with assistant-only loss and a 20k thinking context.
# Compare with matching loss/length settings to isolate harmonization effects.
# --lr is scaled 10x for LoRA; job names use the effective rate.
set -euo pipefail
cd /home/vijayv/forger/post_data_bench/coding/

DATA=/home/vijayv/forger/post_data_bench/coding/.hidden_agents/vijay_manual_8_23_26/full_long_fixed

submit () {  # $1 job name, $2 pre-scaling lr, $3 max-length, $4... extra flags
  local job=$1 lr=$2 maxlen=$3; shift 3
  sbatch --partition=general --job-name="$job" --gres=gpu:4 --cpus-per-gpu=8 \
    --constrain="L40S|A100_80GB|A100_40GB|H100|H200" \
    --mem-per-gpu=90G --time 30:00:00 --output="logs/${job}.out" \
    --error="logs/${job}.err" --mail-user=vijayv@andrew.cmu.edu \
    --mail-type=ALL run_model_training_and_evaluation.sh \
      --agent-data-dir "$DATA" --model Qwen/Qwen3-4B --split val \
      --grad-accum 32 --max-length "$maxlen" --lr "$lr" --assistant-only-loss "$@"
}

# ---- non-thinking (max-length 4096) -----------------------------------------
submit lc-manf-fix-lr5e6-nothink 5e-7 4096 --no-thinking
submit lc-manf-fix-lr1e5-nothink 1e-6 4096 --no-thinking
submit lc-manf-fix-lr2e5-nothink 2e-6 4096 --no-thinking

# ---- thinking (max-length 20000) --------------------------------------------
submit lc-manf-fix-lr3e5-think 3e-6 20000
submit lc-manf-fix-lr5e5-think 5e-6 20000
submit lc-manf-fix-lr1e4-think 1e-5 20000
