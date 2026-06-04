#!/bin/bash
#SBATCH --job-name=qwen3-gpqa
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Babel batch submission: evaluate Qwen3-1.7B on GPQA Diamond via the
# Forger Inspect bridge. See scripts/README.md for one-time setup
# (venv, GPQA license acceptance, huggingface-cli login).

set -euo pipefail

# Put HF caches on big storage; /home is only 100GB. Trigger AutoFS by
# stat'ing the full path before first use.
export HF_HOME="/data/user_data/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export INSPECT_DISPLAY=plain
stat "/data/user_data/${USER}" >/dev/null 2>&1 || true
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" logs

# Activate the project env (created during one-time setup).
# shellcheck disable=SC1091
source "/data/user_data/${USER}/envs/forger/bin/activate"

echo "node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} job=${SLURM_JOB_ID:-local}"
nvidia-smi -L || true

python scripts/eval_qwen3_gpqa.py \
    --model "Qwen/Qwen3-1.7B" \
    --task "inspect_evals/gpqa_diamond" \
    --max-tokens 4096 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 8192 \
    --log-dir "logs/inspect-${SLURM_JOB_ID:-local}" \
    --json-output "logs/qwen3-gpqa-${SLURM_JOB_ID:-local}.json"
