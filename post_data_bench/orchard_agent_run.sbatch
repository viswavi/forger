#!/usr/bin/env bash
#SBATCH --job-name=forger-local
#SBATCH --partition=flame
#SBATCH --qos=flame-16gpu-b_qos
#SBATCH --account=gneubig
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=24:00:00
set -euo pipefail
export PATH="$HOME/.local/apptainer/bin:$PATH"
# sbatch passes on the submitting shell's environment, and ~/.bashrc points
# these at /project and gcsfuse; outputs, scratch and RL output stay on the
# node's local disk and go to GCS (pre-eval sync and results.tgz).
unset AGENT_MODEL_OUTPUT_DIR AGENT_RL_OUTPUT_DIR AGENT_SCRATCH_ROOT
# Submit with PDB_REPO, PDB_LAUNCHER_PYTHON, AGENT_HF_HOME, PDB_BENCHMARK_DIR
# set, and --env-file /path/to/credentials.
: "${PDB_REPO:?Set the isolated checkout path}"
if [[ -z "${PDB_LAUNCHER_PYTHON:-}" ]]; then
  : "${PDB_RELEASE:?Set the prepared release identifier}"
  RUNTIME_RELEASE="${PDB_RUNTIME_RELEASE:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("runtime_release", sys.argv[2]))' "$PDB_REPO/DEPLOYMENT.json" "$PDB_RELEASE")}"
  RUNTIME="/tmp/$USER/forger-runtime-$RUNTIME_RELEASE"
  mkdir -p "$RUNTIME"
  # A per-job archive name: node-local /tmp outlives jobs, and gsutil keeps
  # resumable-download state in $HOME keyed by destination path, so a fixed
  # name could resume a stale partial file (crc32c mismatch, job 161764).
  archive="$RUNTIME/launcher.${SLURM_JOB_ID}.tgz"
  source_uri="gs://cmu-gpucloud-vijayv/flame_port/releases/$RUNTIME_RELEASE/launcher.tgz"
  rm -rf "$RUNTIME"/launcher*.tgz "$RUNTIME/launcher"
  gsutil cp "$source_uri" "$archive" || { rm -f "$archive"; gsutil cp "$source_uri" "$archive"; }
  tar -xzf "$archive" -C "$RUNTIME"
  rm -f "$archive"
  export PDB_LAUNCHER_PYTHON="$RUNTIME/launcher/bin/python"
fi
export AGENT_SCRATCH_ROOT="${AGENT_SCRATCH_ROOT:-/tmp/$USER/agent-${SLURM_JOB_ID}}"
export AGENT_MODEL_OUTPUT_DIR="${AGENT_MODEL_OUTPUT_DIR:-$AGENT_SCRATCH_ROOT/outputs}"
if [[ -z "${AGENT_HF_HOME:-}" ]]; then
  export AGENT_HF_HOME="$AGENT_SCRATCH_ROOT/hf_cache"
  mkdir -p "$AGENT_HF_HOME/hub"
  model_src="${PDB_MODEL_CACHE_SOURCE:-/project/flame/vijayv/hf_cache}/hub/models--Qwen--Qwen3-4B"
  cp -a "$model_src" "$AGENT_HF_HOME/hub/"
  # A shared-blob cache links blobs/<sha> to ../../blobs/xx/<sha>, outside the
  # model directory, which the sandbox bind refuses. Copy the blobs as files;
  # the snapshot -> blobs links stay relative.
  rm -rf "$AGENT_HF_HOME/hub/models--Qwen--Qwen3-4B/blobs"
  cp -rL "$model_src/blobs" "$AGENT_HF_HOME/hub/models--Qwen--Qwen3-4B/blobs"
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
# Outputs go to GCS as an incremental, parallel copy: every PDB_SYNC_MINUTES
# during the run (node loss or SIGKILL lose at most that much), before the
# secret-test evaluation, and at exit. No tar.gz: weights barely compress, and
# one-core gzip of ~1 TB took most of a day (jobs 161781/161782).
: "${PDB_ARCHIVE_URI:?Set a unique GCS destination for the outputs of this job}"
sync_to_gcs() {
  local ok=0 d name
  gsutil -q -m rsync -r "$AGENT_MODEL_OUTPUT_DIR" "$PDB_ARCHIVE_URI/outputs" || ok=1
  # The sandbox's workspace and /tmp (agents keep logs there) as small tarballs:
  # tar keeps symlinks and skips sockets, which gsutil cannot copy.
  for d in "$AGENT_SCRATCH_ROOT"/agent_workspace_*/{workspace,rl_out,tmp}; do
    [[ -d "$d" ]] || continue
    name=sandbox_$(basename "$d").tgz
    tar --ignore-failed-read -czf "$AGENT_SCRATCH_ROOT/$name" -C "$(dirname "$d")" "$(basename "$d")" 2>/dev/null
    gsutil -q cp "$AGENT_SCRATCH_ROOT/$name" "$PDB_ARCHIVE_URI/$name" || ok=1
  done
  return "$ok"
}
stop_sync_loop() {  # the loop and any copy it has in flight
  [[ -n "${SYNC_LOOP:-}" ]] || return 0
  pkill -P "$SYNC_LOOP" 2>/dev/null || true
  kill "$SYNC_LOOP" 2>/dev/null || true
  SYNC_LOOP=""
}
archive_outputs() {
  code=$?
  trap - EXIT
  stop_sync_loop
  if sync_to_gcs; then echo ">>> final sync done: $PDB_ARCHIVE_URI/"; else echo "WARNING: final sync to $PDB_ARCHIVE_URI/ failed" >&2; fi
  exit "$code"
}
mkdir -p "$AGENT_SCRATCH_ROOT"
trap archive_outputs EXIT
( while sleep "$(( ${PDB_SYNC_MINUTES:-30} * 60 ))"; do sync_to_gcs || true; done ) &
SYNC_LOOP=$!
# Freeze the submission without evaluating, copy everything to GCS, then run
# secret-test evaluation on the frozen bundle, so the agent's 15 hours of work
# are safe before the evaluation starts.
FINAL_BUNDLE="$AGENT_MODEL_OUTPUT_DIR/coding/final_bundle_${SLURM_JOB_ID}"
PREPARE_ONLY=0
for arg in "$@"; do [[ "$arg" == --prepare-final-only ]] && PREPARE_ONLY=1; done
"$PDB_LAUNCHER_PYTHON" launch_agent.py --backend local --task coding \
  --prompt coding/FORGER_PROMPT.txt --timeout-hours 15 "$@" \
  --prepare-final-only --final-output-dir "$FINAL_BUNDLE"
stop_sync_loop
if sync_to_gcs; then
  echo ">>> pre-eval sync done: $PDB_ARCHIVE_URI/"
else
  echo "WARNING: pre-eval sync to $PDB_ARCHIVE_URI/ failed; evaluating anyway" >&2
fi
if [[ "$PREPARE_ONLY" == 0 ]]; then
  "$PDB_HOST_PYTHON" -m rl.agent_training run-final --bundle "$FINAL_BUNDLE"
fi
