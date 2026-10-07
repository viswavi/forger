#!/usr/bin/env bash
# Build a versioned benchmark from externally supplied frozen generation artifacts.
# Missing paid generations require --allow-generation. Downloads resolve the HF
# revision to an immutable commit. Prefer --output-dir for a separate build.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SPLIT_ARGS=()
GOLD_ARGS=()
PERTURB_ARGS=()
ALLOW_GENERATION=0
REVISION=main
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) sed -n '2,4p' "$0"; echo 'Options: --output-dir DIR --force --source-revision REV --allow-generation --gold-arg ARG --perturbation-arg ARG'; exit 0 ;;
    --force) SPLIT_ARGS+=(--force); shift ;;
    --output-dir) SPLIT_ARGS+=(--output-dir "$2"); shift 2 ;;
    --source-revision) REVISION="$2"; shift 2 ;;
    --allow-generation) ALLOW_GENERATION=1; shift ;;
    --gold-arg) GOLD_ARGS+=("$2"); shift 2 ;;
    --perturbation-arg) PERTURB_ARGS+=("$2"); shift 2 ;;
    --generation-arg) [[ "$2" != --frozen-file* ]] || { echo 'Use --gold-file / --perturbed-file for artifact paths' >&2; exit 2; }; GOLD_ARGS+=("$2"); PERTURB_ARGS+=("$2"); shift 2 ;;
    --gold-file) GOLD_ARGS+=(--frozen-file "$2"); SPLIT_ARGS+=(--gold "$2"); shift 2 ;;
    --perturbed-file) PERTURB_ARGS+=(--frozen-file "$2"); SPLIT_ARGS+=(--perturbed "$2"); shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if ((ALLOW_GENERATION == 0)); then GOLD_ARGS+=(--no-generation); PERTURB_ARGS+=(--no-generation); fi
bash ../setup_dependencies.sh
python download_sources.py --revision "$REVISION"
python gold_generation.py "${GOLD_ARGS[@]}"
python perturb.py "${PERTURB_ARGS[@]}"
python make_splits.py "${SPLIT_ARGS[@]}"
