#!/usr/bin/env bash
#SBATCH --job-name=forger-prepare
#SBATCH --partition=flame
#SBATCH --qos=flame-16gpu-b_qos
#SBATCH --account=gneubig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
: "${PDB_REPO:?Set the versioned source directory}"
: "${PDB_RELEASE:?Set the source bundle identifier}"
export PATH="$HOME/.local/apptainer/bin:$PATH"
export SIF_NAME=openhands-forger-flame-v3-632f768.sif
source "$HOME/stage_sif.sh"
ROOT="/tmp/$USER/forger-runtime-$PDB_RELEASE"
mkdir -p "$ROOT" "$ROOT/pip-cache"
export PIP_CACHE_DIR="$ROOT/pip-cache"
python3 -m venv "$ROOT/launcher"
"$ROOT/launcher/bin/python" -m pip install \
  openhands-sdk==1.54.0 openhands-agent-server==1.54.0 \
  openhands-tools==1.54.0 openhands-workspace==1.54.0 \
  litellm==1.105.0 python-dotenv
"$ROOT/launcher/bin/python" -m pip freeze > "$ROOT/launcher-freeze.txt"
# Preserve the venv at the same absolute prefix when restored on another node.
tar -C "$ROOT" -czf "$ROOT/launcher.tgz" launcher launcher-freeze.txt
GSUTIL="$(command -v gsutil)"
DEST="gs://cmu-gpucloud-vijayv/flame_port/releases/$PDB_RELEASE"
"$GSUTIL" -h 'x-goog-if-generation-match:0' cp "$ROOT/launcher.tgz" "$DEST/launcher.tgz"
"$GSUTIL" -h 'x-goog-if-generation-match:0' cp "$PDB_REPO/SOURCE_BUNDLE.tgz" "$DEST/source.tgz"
printf 'release=%s\nnode=%s\nruntime=%s\nimage=%s\narchive=%s\n' \
  "$PDB_RELEASE" "$(hostname)" "$ROOT" "$SIF" "$DEST" > "$ROOT/prepared.txt"
"$GSUTIL" -h 'x-goog-if-generation-match:0' cp "$ROOT/prepared.txt" "$DEST/prepared.txt"
cat "$ROOT/prepared.txt"
