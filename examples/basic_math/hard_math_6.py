"""hard_math_6.py -- lite-gated cross-model consensus (uncapped solvers).

Two fixes over hard_math_5:
  * Solvers are UNCAPPED.  gemini-3.5-flash is a thinking model; a max_tokens cap
    is consumed by hidden reasoning and truncates the visible answer, which broke
    every strong solve in v5.  (Confirmed: capped -> truncated mid-proof; uncapped
    -> full solution + correct 'ANSWER: n'.)
  * Cost control by GATING the expensive strong solver behind a cheap weak-model
    self-consistency check, instead of capping tokens.

Acceptance rule (gold = the agreed integer L):
  1. Solve lite x2.  Keep only problems with lite1 == lite2 == L  (cheap gate).
  2. For survivors, solve flash x N_CONFIRM.  Accept iff every flash solve == L.
     => 2 + N_CONFIRM solves from 2 different models all agree.  Cross-model
        agreement is far stronger than within-model agreement (which produced a
        confident-but-wrong answer in v4).

Default N_CONFIRM=1 (one strong cross-check); set to 2 for extra safety.
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

N_CONFIRM = 1  # number of strong cross-check solves


def verify(problems, lite, strong, n_confirm=N_CONFIRM):
    # Stage 1: cheap weak-model self-consistency gate
    l1 = solve_pass(problems, lite, "lite1")
    l2 = solve_pass(problems, lite, "lite2")
    survivors = []
    for p, a, b in zip(problems, l1, l2):
        if a is not None and a == b:
            p["_L"] = a
            survivors.append(p)

    # Stage 2: strong cross-model confirmation
    flash_votes = {p["pid"]: [] for p in survivors}
    for j in range(n_confirm):
        fj = solve_pass(survivors, strong, f"flash{j+1}")
        for p, v in zip(survivors, fj):
            flash_votes[p["pid"]].append(v)

    accepted = []
    for p in survivors:
        L = p["_L"]
        fv = flash_votes[p["pid"]]
        if fv and all(v == L for v in fv):
            p["gold"] = L
            accepted.append(p)
    return accepted, len(survivors)


def reference_solution(p) -> str:
    gold = p["gold"]
    for tag in ("flash1", "flash2", "lite1", "lite2"):
        s = p.get("_sol_" + tag, "")
        if s and extract_final_int(s) == gold:
            body = s.split("ANSWER:")[0].rstrip()
            return f"{body}\n\n#### {gold}"
    return f"#### {gold}"


def assemble(accepted) -> list[dict]:
    pool = F.by_rouge_l(accepted, field="question", threshold=0.55)
    pool = F.by_n_gram_overlap(pool, field="question", n=4, threshold=0.5)
    out = []
    for p in pool:
        out.append({
            "question": p["question"],
            "answer": reference_solution(p),
            "final_answer": p["gold"],
            "topic": p["topic"],
            "subtopic": p["subtopic"],
        })
    return out


def run_batch(n_seeds, seed, n_confirm=N_CONFIRM):
    writer = make_model(WRITER_ID)
    lite = make_model(LITE_SOLVER_ID, max_tokens=None)
    strong = make_model(STRONG_SOLVER_ID, max_tokens=None)
    seeds = sample_seeds(n_seeds, seed=seed)
    problems = generate_problems(seeds, writer)
    problems = F.by_rouge_l(problems, field="question", threshold=0.6)
    for i, p in enumerate(problems):
        p["pid"] = i
    accepted, n_surv = verify(problems, lite, strong, n_confirm)

    def _c(m):
        return m.last_usage["total_cost"] if m.last_usage else 0.0
    cost = _c(writer) + _c(lite) + _c(strong)
    return problems, accepted, n_surv, cost


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    out_name = sys.argv[3] if len(sys.argv) > 3 else "inspect_v6.jsonl"

    problems, accepted, n_surv, cost = run_batch(n_seeds, seed)
    records = assemble(accepted)
    print(f"[gen->dedup] {len(problems)} | lite-gate survivors {n_surv} | "
          f"accepted {len(accepted)} | final {len(records)}")
    print(f"[cost] ${cost:.4f}  (~${cost/max(1,len(records)):.4f}/record)  "
          f"[topics] {dict(Counter(r['topic'] for r in records))}")

    out = os.path.join(os.path.dirname(__file__), "..", "hard_math", out_name)
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[write] {len(records)} -> {out}")


if __name__ == "__main__":
    main()
