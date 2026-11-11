#!/bin/bash
# Build the deterministic validation/secret-test split from the pinned dataset file.
# Existing test outputs require --force; --output-dir creates a new version.
set -e
cd "$(dirname "$0")"

DATA="${HB_DATA:-raw/healthbench_professional_eval.jsonl}"
if [[ ! -f "$DATA" ]]; then
  echo ">>> [1/2] downloading the dataset (pinned revision) to raw/"
  python3 -c "from huggingface_hub import hf_hub_download as d; import make_splits as m; d(m.DATASET, 'healthbench_professional_eval.jsonl', repo_type='dataset', revision=m.REVISION, local_dir='raw')"
fi
echo ">>> [2/2] validation_tasks.jsonl / .hidden/test_tasks.jsonl (make_splits.py)"
python3 make_splits.py --data "$DATA" "$@"

echo ">>> done."
