"""Repair the dataset's `answer` field from cached solves (ZERO new API spend).

Bug being fixed: `hard_math_9`'s cost-optimized verifier stored solver solutions
under tags `c9flash1/c9lite1/c9lite2`, but `reference_solution()` only looked for
the old `v7*` tags, so every cost-optimized record fell back to a bare
`#### <answer>` with no worked solution.  The full solutions are in the @op cache;
this script replays the (cached) solves with the correct tag lookup, builds a
question -> full-solution map, and patches the stub answers in place.  A litellm
cost callback aborts if anything turns into a live call (budget guard).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
os.environ.setdefault("LITELLM_BASE_URL", "https://cmu.litellm.ai")
os.environ.setdefault(
    "FORGER_CACHE",
    os.path.join(os.path.dirname(__file__), "..", "hard_math", ".forger_cache"),
)

import litellm  # noqa: E402

COST = {"usd": 0.0, "calls": 0}


def _cb(k, r, s, e):
    try:
        c = k.get("response_cost")
        if c is None:
            c = litellm.completion_cost(completion_response=r)
        COST["usd"] += float(c or 0.0)
        COST["calls"] += 1
    except Exception:
        pass


litellm.success_callback = [_cb]

from pyforger import filter as F  # noqa: E402
import hard_math_8 as H8  # noqa: E402
import hard_math_9 as H9  # noqa: E402
from hard_math_1 import extract_final_int  # noqa: E402

DATASET = os.path.join(os.path.dirname(__file__), "..", "hard_math",
                       "hard_math_dataset.jsonl")
BUDGET_GUARD = 0.50  # abort if cached replay unexpectedly costs money


def ref_sol_fixed(p):
    gold = p["gold"]
    for tag in ("c9flash1", "c9lite1", "c9lite2",
                "v7flash1", "v7flash2", "v7flash3", "v7lite1", "v7lite2"):
        s = p.get("_sol_" + tag, "")
        if s and extract_final_int(s) == gold:
            body = s.split("ANSWER:")[0].rstrip()
            return f"{body}\n\n#### {gold}"
    return f"#### {gold}"


def build_map(start_seed, n_chunks, chunk=20):
    """Replay cached cheap-verify chunks; return {question: full_answer}."""
    writer = H8.make(H8.WRITER_ID)
    lite = H8.make(H8.LITE_ID, max_tokens=4096)
    flash = H8.make(H8.FLASH_ID, reasoning_effort="low")
    mp = {}
    for c in range(n_chunks):
        if COST["usd"] > BUDGET_GUARD:
            print(f"[abort] cost guard hit at ${COST['usd']:.3f}")
            break
        seeds = H8.make_seeds(chunk, start_seed + c * chunk)
        try:
            probs = H8.generate_problems(seeds, writer)
            probs = F.by_rouge_l(probs, field="question", threshold=0.6)
            for i, p in enumerate(probs):
                p["pid"] = i
            acc = H9.verify_cheap(probs, lite, flash)
        except Exception as e:
            print(f"[seed {start_seed+c*chunk}] skip: {type(e).__name__}")
            continue
        for p in acc:
            q = p["question"].replace("\\\\", "\\")
            mp[q] = ref_sol_fixed(p)
    return mp


def main():
    rows = [json.loads(l) for l in open(DATASET)]
    stubs = sum(1 for r in rows if len(r["answer"]) < 30)
    print(f"[load] {len(rows)} records, {stubs} stub answers")

    mp = {}
    # cover both production runs' seed ranges (run1: 200000+, run2: 300000+)
    mp.update(build_map(200000, 40))
    print(f"[map] run1 {len(mp)} solutions, cost ${COST['usd']:.4f}")
    mp.update(build_map(300000, 40))
    print(f"[map] total {len(mp)} solutions, cost ${COST['usd']:.4f} ({COST['calls']} calls)")

    fixed = 0
    for r in rows:
        if len(r["answer"]) < 30:
            full = mp.get(r["question"])
            if full and len(full) >= 30:
                r["answer"] = full
                fixed += 1
    remaining = sum(1 for r in rows if len(r["answer"]) < 30)
    print(f"[patch] filled {fixed} solutions; {remaining} still stubs")

    with open(DATASET, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ls = sorted(len(r["answer"]) for r in rows)
    print(f"[write] answer length: min {ls[0]} median {ls[len(ls)//2]} max {ls[-1]}")


if __name__ == "__main__":
    main()
