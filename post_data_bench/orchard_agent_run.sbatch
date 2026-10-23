#!/usr/bin/env bash
#SBATCH --job-name=forger-local
#SBATCH --partition=flame
#SBATCH --qos=flame-16gpu-b_qos
#SBATCH --account=gneubig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
set -euo pipefail
export PATH="$HOME/.local/apptainer/bin:$PATH"
# Submit with PDB_REPO, PDB_LAUNCHER_PYTHON, AGENT_MODEL_OUTPUT_DIR,
# AGENT_HF_HOME, PDB_BENCHMARK_DIR set, and --env-file /path/to/credentials.
: "${PDB_REPO:?Set the isolated checkout path}"
if [[ -z "${PDB_LAUNCHER_PYTHON:-}" ]]; then
  : "${PDB_RELEASE:?Set the prepared release identifier}"
  RUNTIME_RELEASE="${PDB_RUNTIME_RELEASE:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("runtime_release", sys.argv[2]))' "$PDB_REPO/DEPLOYMENT.json" "$PDB_RELEASE")}"
  RUNTIME="/tmp/$USER/forger-runtime-$RUNTIME_RELEASE"
  mkdir -p "$RUNTIME"
  gsutil cp "gs://cmu-gpucloud-vijayv/flame_port/releases/$RUNTIME_RELEASE/launcher.tgz" "$RUNTIME/launcher.tgz"
  tar -xzf "$RUNTIME/launcher.tgz" -C "$RUNTIME"
  export PDB_LAUNCHER_PYTHON="$RUNTIME/launcher/bin/python"
fi
export AGENT_SCRATCH_ROOT="${AGENT_SCRATCH_ROOT:-/tmp/$USER/agent-${SLURM_JOB_ID}}"
export AGENT_MODEL_OUTPUT_DIR="${AGENT_MODEL_OUTPUT_DIR:-$AGENT_SCRATCH_ROOT/outputs}"
if [[ -z "${AGENT_HF_HOME:-}" ]]; then
  export AGENT_HF_HOME="$AGENT_SCRATCH_ROOT/hf_cache"
  mkdir -p "$AGENT_HF_HOME/hub"
  cp -a "${PDB_MODEL_CACHE_SOURCE:-/project/flame/vijayv/hf_cache}/hub/models--Qwen--Qwen3-4B" "$AGENT_HF_HOME/hub/"
fi
: "${PDB_BENCHMARK_DIR:?Set the verified benchmark build}"
[[ "${1:-}" == --env-file && -f "${2:-}" ]] || { echo 'Specify --env-file PATH first' >&2; exit 2; }
set -a
source "$2"
set +a
shift 2
: "${LLM_AGENT_API_KEY:?Set the separate agent key}"
: "${LITELLM_DATAGEN_API_KEY:?Set the generation key}"
: "${LITELLM_JUDGE_API_KEY:?Set the separate judge key}"
[[ "$LLM_AGENT_API_KEY" != "$LITELLM_DATAGEN_API_KEY" && \
   "$LITELLM_JUDGE_API_KEY" != "$LITELLM_DATAGEN_API_KEY" && \
   "$LLM_AGENT_API_KEY" != "$LITELLM_JUDGE_API_KEY" ]] || { echo 'Use distinct agent, generation and judge keys' >&2; exit 2; }
export SIF_NAME="${SIF_NAME:-openhands-forger-flame-v3-632f768.sif}"
source "$HOME/stage_sif.sh"
export AGENT_SIF="$SIF"
export AGENT_SCRATCH_ROOT="${AGENT_SCRATCH_ROOT:-/tmp/$USER/agent-${SLURM_JOB_ID}}"
export AGENT_GPU_DESC='8 H100 80GB GPUs'
export AGENT_MODEL_OUTPUT_DIR AGENT_HF_HOME PDB_BENCHMARK_DIR
export PDB_HOST_PYTHON="$PDB_REPO/post_data_bench/local_host_python.sh"
cd "$PDB_REPO/post_data_bench"
# Archive normal and failed exits; scheduler SIGKILL/node loss still needs recovery.
: "${PDB_ARCHIVE_URI:?Set a unique GCS destination for this job's outputs}"
archive_outputs() {
  code=$?
  trap - EXIT
  tar -C "$AGENT_MODEL_OUTPUT_DIR" -czf "$AGENT_SCRATCH_ROOT/results.tgz" .
  gsutil -h 'x-goog-if-generation-match:0' cp "$AGENT_SCRATCH_ROOT/results.tgz" "$PDB_ARCHIVE_URI/results.tgz" || exit 1
  exit "$code"
}
mkdir -p "$AGENT_SCRATCH_ROOT"
trap archive_outputs EXIT
"$PDB_LAUNCHER_PYTHON" launch_agent.py --backend local --task coding \
  --prompt coding/FORGER_PROMPT.txt --timeout-hours 15 "$@"
