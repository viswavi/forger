#!/usr/bin/env bash
# Assistant-only LoRA sweep; otherwise matches the all-token-loss baseline.
# --lr is scaled 10x by train_sft.py; job names use the effective rate.
# Distinct job/output suffixes prevent collisions with the baseline.
set -euo pipefail

cd /home/vijayv/forger/post_data_bench/coding/

DATA_DIR=/home/vijayv/forger/post_data_bench/coding/.hidden_agents/vijay_manual_8_23_26/full_long

submit () {  # $1 = job name, $2 = pre-scaling lr
  sbatch --partition=general --job-name="$1" --gres=gpu:4 --cpus-per-gpu=8 \
    --constrain="L40S|A100_80GB|A100_40GB|H100|H200" \
    --mem-per-gpu=90G --time 30:00:00 --output="logs/$1.out" \
    --error="logs/$1.err" --mail-user=vijayv@andrew.cmu.edu \
    --mail-type=ALL run_model_training_and_evaluation.sh \
      --agent-data-dir "$DATA_DIR" \
      --model Qwen/Qwen3-4B --split val \
      --grad-accum 32 --max-length 4096 --lr "$2" \
      --no-thinking --assistant-only-loss
}

submit lc-manf-lr1e5-nothink-aol 1e-6
submit lc-manf-lr2e5-nothink-aol 2e-6
submit lc-manf-lr3e5-nothink-aol 3e-6
