"""hard_math_4.py -- near-final pipeline.

Changes from hard_math_3:
  * Tier 2 relaxed to pure 2/2 strong-solver (flash) unanimity (the lite-membership
    cross-check in v3 was too strict and accepted 0).  Two independent strong solves
    agreeing is itself a strong correctness signal and rejects ill-posed problems
    (their solves disagree).
  * Final cross-set diversity pass (rouge-l + n-gram) over the accepted pool, so we
    don't ship semantically near-duplicate problems generated across batches.
  * Assembles the final training record: question + full worked solution rewritten
    to end with the GSM8K-style '#### <answer>' marker, plus the integer answer and
    provenance.

Usage:
    python hard_math_4.py            # default validation batch
    python hard_math_4.py 40 200     # n_seeds seed
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
os.environ.setdefault("LITELLM_API_KEY", "sk-i1GxNCIGJnywAvUfy5iw1Q")
os.environ.setdefault("LITELLM_BASE_URL", "https://cmu.litellm.ai")
os.environ.setdefault(
    "FORGER_CACHE",
    os.path.join(os.path.dirname(__file__), "..", "hard_math", ".forger_cache"),
)

from pyforger import filter as F  # noqa: E402

from hard_math_1 import extract_final_int  # noqa: E402
from hard_math_2 import sample_seeds  # noqa: E402
from hard_math_3 import (  # noqa: E402
    make_model, generate_problems, solve_pass,
    WRITER_ID, LITE_SOLVER_ID, STRONG_SOLVER_ID,
)


def verify(problems, lite, strong):
    """Tier1: 3x flash-lite unanimous.  Tier2: 2x flash unanimous."""
    l1 = solve_pass(problems, lite, "lite1")
    l2 = solve_pass(problems, lite, "lite2")
    l3 = solve_pass(problems, lite, "lite3")
    accepted, unresolved = [], []
    for p, a, b, c in zip(problems, l1, l2, l3):
        p["_lite_votes"] = [v for v in (a, b, c) if v is not None]
        if a is not None and a == b == c:
            p["gold"], p["tier"] = a, 1
            accepted.append(p)
        else:
            unresolved.append(p)
    if unresolved:
        f1 = solve_pass(unresolved, strong, "flash1")
        f2 = solve_pass(unresolved, strong, "flash2")
        for p, a, b in zip(unresolved, f1, f2):
            if a is not None and a == b:
                p["gold"], p["tier"] = a, 2
                accepted.append(p)
    return accepted


def reference_solution(p) -> str:
    """Pick a full solution matching gold and rewrite its tail to '#### <gold>'."""
    gold = p["gold"]
    for tag in ("flash1", "flash2", "lite1", "lite2", "lite3"):
        s = p.get("_sol_" + tag, "")
        if s and extract_final_int(s) == gold:
            # normalize the ANSWER line to the GSM8K-style marker
            body = s.split("ANSWER:")[0].rstrip()
            return f"{body}\n\n#### {gold}"
    return f"#### {gold}"


def assemble(accepted) -> list[dict]:
    """Final cross-set dedup + record assembly."""
    pool = [{**p, "question": p["question"]} for p in accepted]
    pool = F.by_rouge_l(pool, field="question", threshold=0.55)
    pool = F.by_n_gram_overlap(pool, field="question", n=4, threshold=0.5)
    records = []
    for p in pool:
        records.append({
            "question": p["question"],
            "answer": reference_solution(p),
            "final_answer": p["gold"],
            "topic": p["topic"],
            "subtopic": p["subtopic"],
            "difficulty_tier": p["tier"],  # 1=lite-solvable, 2=needs strong solver
        })
    return records


def run_batch(n_seeds, seed):
    writer = make_model(WRITER_ID)
    lite = make_model(LITE_SOLVER_ID, max_tokens=2048)
    strong = make_model(STRONG_SOLVER_ID, max_tokens=2048)

    seeds = sample_seeds(n_seeds, seed=seed)
    problems = generate_problems(seeds, writer)
    problems = F.by_rouge_l(problems, field="question", threshold=0.6)
    for i, p in enumerate(problems):
        p["pid"] = i
    accepted = verify(problems, lite, strong)

    def _cost(m):
        u = m.last_usage
        return u["total_cost"] if u else 0.0
    cost = _cost(writer) + _cost(lite) + _cost(strong)
    return problems, accepted, cost


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    problems, accepted, cost = run_batch(n_seeds, seed)
    records = assemble(accepted)
    t1 = sum(1 for p in accepted if p["tier"] == 1)
    t2 = sum(1 for p in accepted if p["tier"] == 2)
    print(f"[gen->dedup] {len(problems)} | [accepted] {len(accepted)} "
          f"(tier1={t1} tier2={t2}) | [after final dedup] {len(records)}")
    print(f"[topics] {dict(Counter(r['topic'] for r in records))}")
    print(f"[cost] uncached this run: ${cost:.4f}")

    out = os.path.join(os.path.dirname(__file__), "..", "hard_math", "inspect_v4.jsonl")
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[write] {len(records)} -> {out}")


if __name__ == "__main__":
    main()
