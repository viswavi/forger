#!/bin/bash
#SBATCH --job-name=forger-humaneval
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Babel batch submission: estimate the quality of a Forger candidate dataset
# via HumanEval ICL uplift on Qwen3-1.7B. See scripts/README.md for one-time
# setup (venv, huggingface-cli login, --allow-code-execution rationale).

set -euo pipefail

# Caller must set FORGER_JSONL=/abs/path/to/coding_questions_trl_format.jsonl
: "${FORGER_JSONL:?Set FORGER_JSONL to the candidate-data JSONL path}"

# Put HF caches on big storage; /home is only 100GB. Trigger AutoFS by
# stat'ing the full path before first use.
export HF_HOME="/data/user_data/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
stat "/data/user_data/${USER}" >/dev/null 2>&1 || true
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" logs

# Activate the project env (created during one-time setup).
# shellcheck disable=SC1091
source "/data/user_data/${USER}/envs/forger/bin/activate"

echo "node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} job=${SLURM_JOB_ID:-local}"
nvidia-smi -L || true

python scripts/eval_forger_humaneval.py \
    --forger-jsonl "${FORGER_JSONL}" \
    --model "Qwen/Qwen3-1.7B" \
    --n-demos 20 \
    --n-val 30 \
    --k 4 \
    --n-subsets 5 \
    --seed 0 \
    --max-tokens 1024 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 8192 \
    --timeout-s 10 \
    --allow-code-execution \
    --json-output "logs/forger-humaneval-${SLURM_JOB_ID:-local}.json"
