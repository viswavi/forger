"""hard_math_7.py -- FINAL pipeline: cross-model consensus + well-posedness critic.

Design (validated piece-by-piece in hard_math_1..6):
  * Writer (gemini-3.1-flash-lite) generates AIME/USAMO-style problems conditioned on
    a sampled (topic, subtopic); robust JSON-ish parsing recovers {question, answer}.
  * Solvers use gemini-3.5-flash with reasoning_effort="low".  This is essential:
    the model is a *thinking* model, so an uncapped solve can burn 15k+ tokens
    ($0.14) and a hard max_tokens cap truncates the answer.  "low" effort gives
    correct answers cheaply (~$0.013/solve; verified on AIME-level problems incl.
    log->33, lattice-paths->486, cot^2 sum->19903).
  * Verification = consensus that catches the two error classes we observed:
      - COMPUTATION errors:  require flash self-consistency f1==f2 (==f3 for the
        hard tier).  gold is the STRONG model's value -- this corrected a case
        where flash-lite was confidently wrong (63 vs the true 486).
      - ILL-POSED premises:  a separate referee/critic pass (flash, low effort)
        checks mutual consistency of conditions and validity of claimed answer
        forms.  It caught a complex-valued "rational" sum and an over-constrained
        geometry problem that consensus alone passed.
  * Tiers:  'core'  = flash self-consistent AND a flash-lite solve agrees
            (cross-model, 3 solves / 2 models);
            'hard'  = flash self-consistent but lite disagrees, confirmed by a 3rd
            flash solve (strong 3/3) -- retains genuinely hard problems.
  * Diversity:  early per-batch rouge-l dedup, plus a final cross-set rouge-l +
    n-gram dedup so we don't ship near-duplicate statements.

Usage:
    python hard_math_7.py <n_seeds> <seed> <out_name>
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

from pyforger.acquire import Model  # noqa: E402
from pyforger.alter.generate import generate_from_context  # noqa: E402
from pyforger import filter as F  # noqa: E402

from hard_math_1 import extract_final_int  # noqa: E402
from hard_math_2 import sample_seeds  # noqa: E402
from hard_math_3 import generate_problems, solve_pass, WRITER_ID  # noqa: E402

LITE_ID = "litellm_proxy/gemini/gemini-3.1-flash-lite"
FLASH_ID = "litellm_proxy/gemini/gemini-3.5-flash"


def make(model_id, **kw):
    return Model(model_id, api_key=os.environ["LITELLM_API_KEY"],
                 api_base=os.environ["LITELLM_BASE_URL"], temperature=1.0,
                 num_retries=3, **kw)


# --- well-posedness referee -------------------------------------------------
CRITIC_SYS = (
    "You are a strict competition-math referee checking a problem for "
    "WELL-POSEDNESS. Check explicitly and reason: (1) Are all stated conditions "
    "mutually CONSISTENT -- does a configuration satisfying every condition "
    "simultaneously actually exist? (2) Is every claimed property TRUE -- if it "
    "says a quantity equals m/n with integers, is that quantity actually real and "
    "rational? verify any 'it is known that' assertion. (3) Is the requested answer "
    "UNIQUELY determined and a non-negative integer below 1000? Only output INVALID "
    "if you find a DEFINITE flaw; otherwise VALID. End with exactly "
    "'VERDICT: VALID' or 'VERDICT: INVALID'."
)


def critic_prompt(item):
    return [{"role": "system", "content": CRITIC_SYS},
            {"role": "user", "content": "Problem:\n\n" + item["question"]}]


def verify(problems, lite, flash):
    l1 = solve_pass(problems, lite, "v7lite1")
    l2 = solve_pass(problems, lite, "v7lite2")
    f1 = solve_pass(problems, flash, "v7flash1")
    f2 = solve_pass(problems, flash, "v7flash2")
    accepted, hard_cand = [], []
    for p, a, b, c, d in zip(problems, l1, l2, f1, f2):
        p["_lite_votes"] = [v for v in (a, b) if v is not None]
        if c is not None and c == d:                 # strong self-consistent
            if c in p["_lite_votes"]:                 # cross-model agreement
                p["gold"], p["tier"] = c, "core"
                accepted.append(p)
            else:
                p["_fstrong"] = c
                hard_cand.append(p)
    if hard_cand:
        f3 = solve_pass(hard_cand, flash, "v7flash3")
        for p, e in zip(hard_cand, f3):
            if e is not None and e == p["_fstrong"]:  # strong 3/3
                p["gold"], p["tier"] = p["_fstrong"], "hard"
                accepted.append(p)
    return accepted


def apply_critic(accepted, flash):
    if not accepted:
        return []
    out = generate_from_context(accepted, critic_prompt, flash, samples_per_item=1,
                                temperature=1.0, batch_size=24)
    kept = []
    for p, o in zip(accepted, out):
        txt = o.get("text", "") if o else ""
        verdicts = [m.upper() for m in __import__("re").findall(
            r"VERDICT:\s*(VALID|INVALID)", txt, __import__("re").I)]
        if verdicts and verdicts[-1] == "VALID":
            kept.append(p)
    return kept


def reference_solution(p):
    gold = p["gold"]
    for tag in ("v7flash1", "v7flash2", "v7flash3", "v7lite1", "v7lite2"):
        s = p.get("_sol_" + tag, "")
        if s and extract_final_int(s) == gold:
            body = s.split("ANSWER:")[0].rstrip()
            return f"{body}\n\n#### {gold}"
    return f"#### {gold}"


def assemble(kept):
    pool = F.by_rouge_l(kept, field="question", threshold=0.55)
    pool = F.by_n_gram_overlap(pool, field="question", n=4, threshold=0.5)
    out = []
    for p in pool:
        q = p["question"].replace("\\\\", "\\")  # normalize double-escaped latex
        out.append({
            "question": q,
            "answer": reference_solution(p),
            "final_answer": p["gold"],
            "topic": p["topic"],
            "subtopic": p["subtopic"],
            "difficulty_tier": p["tier"],
        })
    return out


def run(n_seeds, seed):
    writer = make(WRITER_ID)
    lite = make(LITE_ID, max_tokens=4096)
    flash = make(FLASH_ID, reasoning_effort="low")
    seeds = sample_seeds(n_seeds, seed=seed)
    problems = generate_problems(seeds, writer)
    problems = F.by_rouge_l(problems, field="question", threshold=0.6)
    for i, p in enumerate(problems):
        p["pid"] = i
    accepted = verify(problems, lite, flash)
    kept = apply_critic(accepted, flash)
    records = assemble(kept)

    def _c(m):
        return m.last_usage["total_cost"] if m.last_usage else 0.0
    cost = _c(writer) + _c(lite) + _c(flash)
    return problems, accepted, kept, records, cost


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    out_name = sys.argv[3] if len(sys.argv) > 3 else "inspect_v7.jsonl"
    problems, accepted, kept, records, cost = run(n_seeds, seed)
    tc = Counter(p["tier"] for p in accepted)
    print(f"[gen->dedup] {len(problems)} | consensus {len(accepted)} "
          f"(core={tc['core']} hard={tc['hard']}) | post-critic {len(kept)} | "
          f"final {len(records)}")
    print(f"[cost] ${cost:.4f} (${cost/max(1,len(records)):.4f}/record)  "
          f"[topics] {dict(Counter(r['topic'] for r in records))}")
    out = os.path.join(os.path.dirname(__file__), "..", "hard_math", out_name)
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[write] {len(records)} -> {out}")


if __name__ == "__main__":
    main()
