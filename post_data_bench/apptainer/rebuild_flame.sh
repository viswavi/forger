#!/usr/bin/env bash
# Build the combined Slime/TRL image on babel, then transfer it to flame.
# Use allowlisted source files and stage heavy artifacts on /scratch.
# Usage: ./rebuild_flame.sh [--skip-tar]  # reuse TRL tar when iterating
set -euo pipefail
SKIP_TAR=0
while (($#)); do
  case "$1" in
    --skip-tar) SKIP_TAR=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/home/vijayv/miniconda3/envs/trl}"
SLIME_BASE="${SLIME_BASE:-/data/user_data/vijayv/containers/slime.sif}"
BUILD_DIR="${BUILD_DIR:-/scratch/vijayv/apptainer_build}"
TAR_PATH="$BUILD_DIR/trl_env.tar"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default to the combined image so sandbox agents can call Slime.
# For TRL-only: DEF_FILE=$SCRIPT_DIR/openhands-forger-flameA.def ./rebuild_flame.sh
DEF_FILE="${DEF_FILE:-$SCRIPT_DIR/openhands-forger-flame.def}"
SIF_OUT="${SIF_OUT:-/data/user_data/vijayv/containers/$(basename "${DEF_FILE%.def}").sif}"

# $HOME is at its 25 GB quota on flame and tight here too; apptainer's cache and
# tmpdir both need tens of GB. Keep them off $HOME unconditionally.
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/scratch/vijayv/apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/scratch/vijayv/apptainer_tmp}"
mkdir -p "$BUILD_DIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$(dirname "$SIF_OUT")"

echo "=== preflight ==="
# The slime base is only needed by the combined recipe.
if grep -qE '^\s*From:.*slime' "$DEF_FILE" 2>/dev/null; then
  [ -r "$SLIME_BASE" ] || { echo "!! slime base not readable: $SLIME_BASE" >&2; exit 1; }
  echo "  base        : $SLIME_BASE ($(du -h "$SLIME_BASE" | cut -f1))"
else
  echo "  base        : $(grep -E '^\s*From:' "$DEF_FILE" | head -1 | sed 's/^\s*From:\s*//') (pulled)"
fi
[ -d "$CONDA_ENV_PATH" ] || { echo "!! trl env not found: $CONDA_ENV_PATH" >&2; exit 1; }
[ -r "$DEF_FILE" ] || { echo "!! def not found: $DEF_FILE" >&2; exit 1; }
command -v apptainer >/dev/null || { echo "!! apptainer not on PATH" >&2; exit 1; }
avail=$(df -Pm "$BUILD_DIR" | tail -1 | awk '{print $4}')
echo "  trl env     : $CONDA_ENV_PATH ($(du -sh "$CONDA_ENV_PATH" | cut -f1))"
echo "  build dir   : $BUILD_DIR (${avail} MB free)"
echo "  cache/tmp   : $APPTAINER_CACHEDIR , $APPTAINER_TMPDIR"
echo "  recipe      : $DEF_FILE"
echo "  output      : $SIF_OUT"
echo "  apptainer   : $(apptainer --version)"
if [ "$avail" -lt 80000 ]; then
  echo "  !! want >=80000 MB free for tar + build scratch + output; have ${avail}" >&2
  exit 1
fi

# Sanity: the %files allowlist must actually exist, or the build fails late.
echo "=== allowlist ==="
total=0
while IFS= read -r f; do
  [[ "$f" == */trl_env.tar ]] && continue
  [ -r "$f" ] || { echo "!! missing from allowlist: $f" >&2; exit 1; }
  sz=$(stat -c %s "$f"); total=$((total+sz))
  printf '  %8d  %s\n' "$sz" "$f"
done < <(awk '/^%files/{files=1;next} /^%/{files=0} files && NF && $1 !~ /^#/{print $1}' "$DEF_FILE")
echo "  --------  $total bytes total (vs ~96 GB under the old wholesale %files)"

if [ "$SKIP_TAR" -eq 1 ] && [ -f "$TAR_PATH" ]; then
  echo "=== reusing tar: $TAR_PATH ($(du -h "$TAR_PATH" | cut -f1)) ==="
else
  echo "=== tarring $CONDA_ENV_PATH -> $TAR_PATH ==="
  # -C / with a relative path so it extracts to the identical absolute path in
  # %post. No relocation, so the env's shebangs/RPATHs/pip metadata stay valid.
  time tar cf "$TAR_PATH" -C / "${CONDA_ENV_PATH#/}"
  echo "  tar: $(du -h "$TAR_PATH" | cut -f1)"
fi

echo "=== building $SIF_OUT ==="
# Build to a temporary file so a failed build cannot destroy the last good image.
BUILD_RECIPE="$BUILD_DIR/active.def"
python - "$DEF_FILE" "$BUILD_RECIPE" "$TAR_PATH" "$SLIME_BASE" <<'PYTHON'
from pathlib import Path
import sys
source, target, tar, base = sys.argv[1:]
text = Path(source).read_text().replace('/scratch/vijayv/apptainer_build/trl_env.tar', tar)
text = text.replace('From: /data/user_data/vijayv/containers/slime.sif', 'From: '+base)
Path(target).write_text(text)
PYTHON
time apptainer build --fakeroot "$SIF_OUT.pending" "$BUILD_RECIPE"
mv "$SIF_OUT.pending" "$SIF_OUT"

echo
echo "=== built: $SIF_OUT ($(du -h "$SIF_OUT" | cut -f1)) ==="
echo
echo "Smoke test it here before shipping:"
echo "  apptainer exec --fakeroot --compat --nv --no-mount hostfs --no-mount bind-paths \\"
echo "    $SIF_OUT sh -c '/home/vijayv/miniconda3/envs/trl/bin/python -c \"import torch,vllm;print(torch.__version__)\"; \\"
echo "                    /usr/bin/python3 -c \"import torch,sglang;print(torch.__version__)\"'"
echo
echo "Then ship to flame and run flame_apptainer_preflight.sh with SIF= set."

printf "export AGENT_SIF=%q\n" "$SIF_OUT"
