#!/usr/bin/env bash
# Trusted host finalization inside the combined image; never bound into the agent.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${AGENT_SIF:?Set AGENT_SIF to the staged combined image}"
: "${AGENT_MODEL_OUTPUT_DIR:?Set a persistent POSIX output directory}"
: "${AGENT_HF_HOME:?Set the host Hugging Face cache directory}"
: "${PDB_BENCHMARK_DIR:?Set the verified host benchmark directory}"
SCRATCH="${AGENT_SCRATCH_ROOT:-/tmp/$USER}"
mkdir -p "$SCRATCH/host-runtime" "$AGENT_MODEL_OUTPUT_DIR"
# Environment carries credentials; none appear in command arguments.
export APPTAINERENV_HF_HOME="$AGENT_HF_HOME"
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_PATH=/home/vijayv/miniconda3/envs/trl/bin:/usr/local/bin:/usr/bin:/bin
export APPTAINERENV_SLIME_ENV=/usr
export APPTAINERENV_SLIME_DEPS=/opt
export APPTAINERENV_TVM_FFI_CACHE_DIR=/tmp/tvm_ffi
export APPTAINERENV_FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer
export APPTAINERENV_PDB_BENCHMARK_DIR="$PDB_BENCHMARK_DIR"
# --compat clears inherited environment; forward the small host reward allowlist.
# LITELLM_OFFICIAL_JUDGE_*: the host's test grader for the clinical task (the
# agent never gets it); LITELLM_WORKER_BASE_URL is its default gateway.
for name in LITELLM_JUDGE_API_KEY LITELLM_JUDGE_BASE_URL LITELLM_BASE_URL \
    LITELLM_OFFICIAL_JUDGE_KEY LITELLM_OFFICIAL_JUDGE_BASE_URL LITELLM_WORKER_BASE_URL \
    JUDGE_MODEL JUDGE_REASONING_EFFORT JUDGE_TOKEN_UNIT_LIMIT JUDGE_TOKEN_USAGE_FILE \
    JUDGE_MAX_CONSECUTIVE_FAILURES PDB_TASK PDB_RUN_TAG PDB_LOCAL_BASE_MODEL; do
  if [[ -v "$name" ]]; then export "APPTAINERENV_${name}=${!name}"; fi
done
exec apptainer exec --fakeroot --compat --nv --no-mount hostfs --no-mount bind-paths \
  --bind "$REPO:$REPO:ro" \
  --bind "$REPO:/home/vijayv/forger:ro" \
  --bind "$AGENT_MODEL_OUTPUT_DIR:$AGENT_MODEL_OUTPUT_DIR" \
  --bind "$AGENT_HF_HOME:$AGENT_HF_HOME:ro" \
  --bind "$PDB_BENCHMARK_DIR:$PDB_BENCHMARK_DIR:ro" \
  --bind "$SCRATCH/host-runtime:/tmp" \
  --bind /etc/resolv.conf:/etc/resolv.conf:ro \
  --pwd "$REPO/post_data_bench/coding" \
  "$AGENT_SIF" /home/vijayv/miniconda3/envs/trl/bin/python "$@"
