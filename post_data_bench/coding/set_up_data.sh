#!/bin/bash
# One-click, deterministic build of the LiveCodeBench-plus-impossible dataset.
#
# 1. Download the raw LiveCodeBench release shards (skipped if already present).
# 2. gold_generation.py: rejection-sample a verified gold response for every lcb_native problem.
#    Loads already-resolved problems from the checked-in gold_responses.jsonl and only calls the
#    LLM for anything missing -- if that file is complete, this step makes zero network calls.
# 3. perturb.py: same pattern for the difficulty=="hard" "impossible" twins, via the checked-in
#    perturbed_cases.jsonl.
# 4. make_splits.py: pure, deterministic join+split (no network calls, ever) producing the final
#    validation_tasks.jsonl / test_tasks.jsonl.
#
# Re-running this script is always safe and cheap once gold_responses.jsonl/perturbed_cases.jsonl
# are fully populated and committed -- steps 2-3 become pure disk loads and step 4 is a fast,
# deterministic join keyed by a fixed random seed.
#
# LITELLM_API_KEY is only required if gold_responses.jsonl/perturbed_cases.jsonl don't yet cover
# every problem (e.g. the first time this is run, or after the source dataset grows).
set -e
cd "$(dirname "$0")"

echo ">>> [1/4] raw LiveCodeBench release shards"
for f in test.jsonl test2.jsonl test3.jsonl test4.jsonl test5.jsonl test6.jsonl; do
  if [ -f "$f" ]; then
    echo "    $f already present, skipping"
  else
    wget "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/$f"
  fi
done

echo ">>> [2/4] gold responses (gold_generation.py)"
python3 gold_generation.py "$@"

echo ">>> [3/4] impossible perturbations using Claude Sonnet 5 (perturb.py)"
python3 perturb.py "$@"

echo ">>> [4/4] validation_tasks.jsonl / test_tasks.jsonl (make_splits.py)"
python3 make_splits.py

echo ">>> done."
