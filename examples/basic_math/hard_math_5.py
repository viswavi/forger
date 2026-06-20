"""hard_math_5.py -- cross-model consensus verifier (precision-hardened).

Why this exists: brute-force validation of hard_math_4 output revealed that
flash-lite 3x unanimity admits FALSE POSITIVES -- the weak model repeated the
same counting error three times (a balanced-lattice-path problem got gold=63
when the true answer is 486).  Weak-model agreement is not sufficient.

New acceptance rule (gold always comes from the STRONG model):
  Solve lite x2 (cheap) and flash x2 (strong).  Require strong self-consistency
  f1 == f2 == v.  Then:
    * if v is also produced by a lite solve  -> accept, tier='core'
      (4 solves, 2 different models, agree -> very high precision)
    * else  -> one more flash solve; if all three flash agree (3/3 strong
      unanimity) accept with tier='hard'  (keeps genuinely hard problems that
      the weak model can't do, with strong corroboration; gold is the strong
      value, which CORRECTS weak-model errors).
    * if the strong model is not self-consistent -> reject (too unreliable).

This trades some recall (hard problems where even flash is inconsistent are
dropped) for precision -- correctness matters far more than raw difficulty for a
training set.
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
    """Cross-model consensus.  Mutates problems; returns accepted list."""
    l1 = solve_pass(problems, lite, "lite1")
    l2 = solve_pass(problems, lite, "lite2")
    f1 = solve_pass(problems, strong, "flash1")
    f2 = solve_pass(problems, strong, "flash2")

    accepted, hard_candidates = [], []
    for p, a, b, c, d in zip(problems, l1, l2, f1, f2):
        p["_lite_votes"] = [v for v in (a, b) if v is not None]
        p["_flash_votes"] = [v for v in (c, d) if v is not None]
        if c is not None and c == d:           # strong self-consistent
            v = c
            if v in p["_lite_votes"]:           # weak model confirms
                p["gold"], p["tier"] = v, "core"
                accepted.append(p)
            else:
                hard_candidates.append(p)
        # strong not self-consistent -> reject

    if hard_candidates:
        f3 = solve_pass(hard_candidates, strong, "flash3")
        for p, e in zip(hard_candidates, f3):
            v = p["_flash_votes"][0]            # == f1 == f2
            if e is not None and e == v:        # 3/3 strong unanimity
                p["gold"], p["tier"] = v, "hard"
                accepted.append(p)
    return accepted


def reference_solution(p) -> str:
    gold = p["gold"]
    for tag in ("flash1", "flash2", "flash3", "lite1", "lite2"):
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
            "difficulty_tier": p["tier"],
        })
    return out


def run_batch(n_seeds, seed):
    writer = make_model(WRITER_ID)
    lite = make_model(LITE_SOLVER_ID, max_tokens=1536)
    strong = make_model(STRONG_SOLVER_ID, max_tokens=2048)
    seeds = sample_seeds(n_seeds, seed=seed)
    problems = generate_problems(seeds, writer)
    problems = F.by_rouge_l(problems, field="question", threshold=0.6)
    for i, p in enumerate(problems):
        p["pid"] = i
    accepted = verify(problems, lite, strong)

    def _c(m):
        return m.last_usage["total_cost"] if m.last_usage else 0.0
    cost = _c(writer) + _c(lite) + _c(strong)
    return problems, accepted, cost


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    out_name = sys.argv[3] if len(sys.argv) > 3 else "inspect_v5.jsonl"

    problems, accepted, cost = run_batch(n_seeds, seed)
    records = assemble(accepted)
    tc = Counter(p["tier"] for p in accepted)
    print(f"[gen->dedup] {len(problems)} | accepted {len(accepted)} "
          f"(core={tc['core']} hard={tc['hard']}) | final {len(records)}")
    print(f"[accept-rate] {100*len(accepted)//max(1,len(problems))}%  "
          f"[cost] ${cost:.4f}  (~${cost/max(1,len(records)):.4f}/record)")

    out = os.path.join(os.path.dirname(__file__), "..", "hard_math", out_name)
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[write] {len(records)} -> {out}")


if __name__ == "__main__":
    main()
