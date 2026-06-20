"""hard_math_9.py -- production runner with ACCURATE cost tracking + append mode.

Fixes the budget-overrun root cause from the previous run: the cost meter read
``model.last_usage`` once per model per chunk, but each model makes ~4-5 solve
calls per chunk and ``last_usage`` only retains the LAST one -> ~4x undercount,
which is why ~$13 metered actually spent ~$50.

Here we register a litellm success callback that accumulates the true billed
``response_cost`` of every completion (cached @op calls don't hit litellm, so
they cost nothing and aren't counted).  The runner:
  * preloads the existing verified dataset and APPENDS (never overwrites it);
  * checkpoints + globally dedupes every chunk;
  * stops at the target record count or when true spend hits the cap;
  * exits cleanly on a budget-exceeded API error so the key can be swapped.

Usage:
    LITELLM_API_KEY=<key> python hard_math_9.py <target> <chunk> <start_seed> <cap_usd>
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

# --- accurate global cost accumulator via litellm callback -------------------
COST = {"usd": 0.0, "calls": 0}


def _cost_cb(kwargs, response_obj, start_time, end_time):
    try:
        c = kwargs.get("response_cost")
        if c is None:
            c = litellm.completion_cost(completion_response=response_obj)
        COST["usd"] += float(c or 0.0)
        COST["calls"] += 1
    except Exception:
        pass


litellm.success_callback = [_cost_cb]

from pyforger import filter as F  # noqa: E402
import hard_math_8 as H8  # noqa: E402
import hard_math_7 as H7  # noqa: E402
from hard_math_3 import solve_pass  # noqa: E402

HM = os.path.join(os.path.dirname(__file__), "..", "hard_math")
DATASET = os.path.join(HM, "hard_math_dataset.jsonl")


def verify_cheap(problems, lite, flash):
    """Cost-optimized: lite x2 self-consistency gate -> single flash confirm.
    Accept iff lite1==lite2==flash1 (3 solves, 2 models agree). gold = flash value.
    ~halves flash calls vs the flash-x2 + flash-x3 design while keeping cross-model
    consensus (rejects the weak-model-confidently-wrong class)."""
    l1 = solve_pass(problems, lite, "c9lite1")
    l2 = solve_pass(problems, lite, "c9lite2")
    survivors = []
    for p, a, b in zip(problems, l1, l2):
        if a is not None and a == b:
            p["_L"] = a
            survivors.append(p)
    accepted = []
    if survivors:
        f1 = solve_pass(survivors, flash, "c9flash1")
        for p, v in zip(survivors, f1):
            if v is not None and v == p["_L"]:
                p["gold"], p["tier"] = v, "core"
                accepted.append(p)
    return accepted


def process_chunk_cheap(seeds):
    writer = H8.make(H8.WRITER_ID)
    lite = H8.make(H8.LITE_ID, max_tokens=4096)
    flash = H8.make(H8.FLASH_ID, reasoning_effort="low")
    problems = H8.generate_problems(seeds, writer)
    problems = F.by_rouge_l(problems, field="question", threshold=0.6)
    for i, p in enumerate(problems):
        p["pid"] = i
    accepted = verify_cheap(problems, lite, flash)
    kept = H7.apply_critic(accepted, flash)
    recs = []
    for p in kept:
        recs.append({
            "question": p["question"].replace("\\\\", "\\"),
            "answer": H7.reference_solution(p),
            "final_answer": p["gold"],
            "topic": p["topic"], "subtopic": p["subtopic"],
            "difficulty_tier": p["tier"],
        })
    return recs, len(problems), len(accepted)


def load_existing():
    if not os.path.exists(DATASET):
        return []
    return [json.loads(l) for l in open(DATASET)]


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 320
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    start_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 200000
    cap = float(sys.argv[4]) if len(sys.argv) > 4 else 18.0

    all_recs = load_existing()
    print(f"[preload] {len(all_recs)} existing records; target {target}; cap ${cap}",
          flush=True)

    acc = {"cost": 0.0, "in": 0, "out": 0}  # unused legacy meter
    c = 0
    while len(all_recs) < target:
        seeds = H8.make_seeds(chunk, start_seed + c * chunk)
        try:
            recs, n_gen, n_acc = process_chunk_cheap(seeds)
        except Exception as e:
            msg = str(e)
            if "Budget has been exceeded" in msg or "budget" in msg.lower():
                print(f"[chunk {c}] BUDGET EXCEEDED on this key -- stopping. "
                      f"true spend ${COST['usd']:.2f}", flush=True)
                break
            print(f"[chunk {c}] skipped: {type(e).__name__}: {msg[:100]}", flush=True)
            c += 1
            continue
        merged = all_recs + recs
        merged = F.by_rouge_l(merged, field="question", threshold=0.55)
        all_recs = F.by_n_gram_overlap(merged, field="question", n=4, threshold=0.5)
        with open(DATASET, "w") as f:
            for r in all_recs:
                f.write(json.dumps(r) + "\n")
        print(f"[chunk {c}] gen={n_gen} accepted={n_acc} +{len(recs)} | "
              f"TOTAL={len(all_recs)} | true_spend=${COST['usd']:.2f} "
              f"({COST['calls']} calls)", flush=True)
        c += 1
        if COST["usd"] >= cap:
            print(f"[stop] cost cap ${cap} reached (true ${COST['usd']:.2f})", flush=True)
            break

    from collections import Counter
    print(f"[FINAL] {len(all_recs)} records | true_spend=${COST['usd']:.2f}")
    print(f"[tiers] {Counter(r.get('difficulty_tier') for r in all_recs)}")
    print(f"[topics] {Counter(r['topic'] for r in all_recs)}")


if __name__ == "__main__":
    main()
