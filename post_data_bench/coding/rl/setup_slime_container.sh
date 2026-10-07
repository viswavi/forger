#!/usr/bin/env bash
# Fetch and smoke-test the upstream pinned Slime image (~40 GB).
# Its Torch stack is incompatible with TRL; keep training isolated.
# Usage: ./rl/setup_slime_container.sh [--check-only]
set -euo pipefail

SIF_DIR="/data/user_data/vijayv/containers"
SIF="$SIF_DIR/slime.sif"
IMAGE="${SLIME_IMAGE:-}"
FORCE=0
CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --sif) SIF="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    *) echo "!!! unknown arg: $1" >&2; exit 1 ;;
  esac
done

# $HOME is 92% full; apptainer's cache and build scratch both need tens of GB.
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/data/user_data/vijayv/apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/data/user_data/vijayv/apptainer_tmp}"
mkdir -p "$SIF_DIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

if [ "$CHECK_ONLY" -eq 0 ]; then
  [[ "$IMAGE" == *@sha256:* ]] || { echo "Pass --image with an immutable sha256 digest" >&2; exit 1; }
  [ ! -e "$SIF" ] || [ "$FORCE" -eq 1 ] || { echo "Image exists; use --force to replace" >&2; exit 1; }
  echo ">>> pulling $IMAGE -> $SIF"
  apptainer pull --force "$SIF" "$IMAGE"
fi

[ -f "$SIF" ] || { echo "!!! no image at $SIF" >&2; exit 1; }
echo ">>> image: $(ls -lh "$SIF" | awk '{print $5}')"

echo ">>> smoke test (CPU only -- imports and paths, not kernels)"
apptainer exec --cleanenv --bind /data:/data --bind /home/vijayv:/home/vijayv "$SIF" bash -c '
set -e
python3 - <<PY
import importlib, sys
print("python  ", sys.version.split()[0])
failed = False
for mod in ("torch", "sglang", "megatron.core", "slime", "transformers", "ray"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:16s}", getattr(m, "__version__", "(no __version__)"))
    except Exception as e:
        failed = True
        print(f"{mod:16s} FAILED: {type(e).__name__}")
sys.exit(1 if failed else 0)
PY
for d in /root/slime /root/Megatron-LM /root/slime/scripts/models/qwen3-4B.sh; do
  [ -e "$d" ] && echo "ok   $d" || { echo "MISSING $d"; exit 1; }
done
'

echo ">>> checking the study's own code imports inside the image"
CODING_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "$CODING_DIR/rl/impossible_lcb_rm.py" ]; then
apptainer exec --cleanenv --bind /data:/data --bind /home/vijayv:/home/vijayv \
  --env "PYTHONPATH=$CODING_DIR" "$SIF" \
  python3 -c 'import rl.impossible_lcb_rm as m; print("custom_rm resolved:", m.custom_rm)'
fi
echo ">>> done"
