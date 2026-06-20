"""hard_math_3.py -- robust parsing + tiered (cheap->strong) consensus verifier.

Findings from hard_math_2:
  * Writer JSON occasionally unparseable (~22% loss) -> add a robust parser.
  * 2/2-unanimous strong-solver accepts were correct on inspection; the 2/3
    tie-broken accepts were the bad tail (one was a provably DIVERGENT, ill-posed
    series).  => require UNANIMITY, never tie-break.
  * Always-flash verification is too expensive (~$0.05/solve).  => tier it:
      Tier 1 (cheap):  3x flash-lite.  If all three agree -> accept (gold = value).
                       Weak models tend to DISAGREE when guessing, so unanimity of
                       three independent weak samples is a strong, cheap signal.
      Tier 2 (strong): for the rest, 2x flash.  Accept iff the two flash solves
                       agree with each other AND the value was also produced by at
                       least one flash-lite pass (cross-model corroboration).

This iteration measures tier pass-rates, correctness and cost on a fresh batch.
"""
from __future__ import annotations

import json
import os
import random
import re
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
from hard_math_2 import sample_seeds, writer_prompt  # noqa: E402

WRITER_ID = "litellm_proxy/gemini/gemini-3.1-flash-lite"
LITE_SOLVER_ID = "litellm_proxy/gemini/gemini-3.1-flash-lite"
STRONG_SOLVER_ID = "litellm_proxy/gemini/gemini-3.5-flash"


def make_model(model_id: str, max_tokens: int | None = None) -> Model:
    return Model(
        model_id,
        api_key=os.environ["LITELLM_API_KEY"],
        api_base=os.environ["LITELLM_BASE_URL"],
        temperature=1.0,
        max_tokens=max_tokens,
        num_retries=3,
    )


# --- robust JSON-ish parsing -----------------------------------------------
def robust_parse(text: str) -> dict | None:
    """Recover {question, answer} from a possibly fenced / noisy LLM response."""
    if not text:
        return None
    t = text.strip()
    # strip markdown fences
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t.strip())
    # try direct, then the outermost {...} block
    candidates = [t]
    first, last = t.find("{"), t.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(t[first : last + 1])
    for c in candidates:
        try:
            d = json.loads(c)
            if isinstance(d, dict) and "question" in d and "answer" in d:
                return d
        except (json.JSONDecodeError, TypeError):
            continue
    # last-ditch: regex the two fields out
    q = re.search(r'"question"\s*:\s*"(.*?)"\s*,\s*"answer"', t, re.S)
    a = re.search(r'"answer"\s*:\s*"?([^"\n}]+)"?', t, re.S)
    if q and a:
        return {"question": q.group(1), "answer": a.group(1)}
    return None


def solver_prompt(item: dict):
    return [
        {
            "role": "system",
            "content": (
                "You are a meticulous competition mathematician. Solve the problem "
                "correctly. Show only the essential steps -- be concise. The final "
                "answer is a non-negative integer below 1000. End with a line exactly "
                "of the form 'ANSWER: <integer>'. If the problem is ill-posed, "
                "ambiguous, or has no such integer answer, end with 'ANSWER: NONE'."
            ),
        },
        {"role": "user", "content": f"Solve this problem:\n\n{item['question']}"},
    ]


def solve_pass(problems, model, tag, batch_size=24):
    items = [{"question": p["question"], "_tag": tag} for p in problems]
    out = generate_from_context(
        items, solver_prompt, model, samples_per_item=1,
        temperature=1.0, batch_size=batch_size,
    )
    res = []
    for p, o in zip(problems, out):
        txt = o.get("text", "") if o else ""
        p.setdefault("_sol_" + tag, txt)
        res.append(extract_final_int(txt))
    return res


def generate_problems(seeds, writer):
    """Generate with robust raw-text parsing (output_schema=None)."""
    out = generate_from_context(
        seeds, writer_prompt, writer, samples_per_item=1,
        output_schema=None, temperature=1.0, batch_size=12,
    )
    problems = []
    for s, o in zip(seeds, out):
        d = robust_parse(o.get("text", "")) if o else None
        if not d or not str(d.get("question", "")).strip():
            continue
        problems.append({
            "pid": len(problems), "topic": s["topic"], "subtopic": s["subtopic"],
            "question": str(d["question"]).strip(),
            "writer_answer": extract_final_int(str(d.get("answer", ""))),
        })
    return problems


def verify(problems, lite, strong):
    """Tiered consensus verification. Mutates problems; returns accepted list."""
    # Tier 1: 3x flash-lite unanimity
    l1 = solve_pass(problems, lite, "lite1")
    l2 = solve_pass(problems, lite, "lite2")
    l3 = solve_pass(problems, lite, "lite3")
    accepted, unresolved = [], []
    for p, a, b, c in zip(problems, l1, l2, l3):
        p["_lite_votes"] = [v for v in (a, b, c) if v is not None]
        if a is not None and a == b == c:
            p["gold"] = a
            p["tier"] = 1
            accepted.append(p)
        else:
            unresolved.append(p)

    # Tier 2: 2x flash unanimity + cross-model corroboration with a lite vote
    if unresolved:
        f1 = solve_pass(unresolved, strong, "flash1")
        f2 = solve_pass(unresolved, strong, "flash2")
        for p, a, b in zip(unresolved, f1, f2):
            if a is not None and a == b and a in p["_lite_votes"]:
                p["gold"] = a
                p["tier"] = 2
                accepted.append(p)
    return accepted


def main(n_seeds=20, seed=100):
    writer = make_model(WRITER_ID)
    lite = make_model(LITE_SOLVER_ID, max_tokens=2048)
    strong = make_model(STRONG_SOLVER_ID, max_tokens=2048)

    seeds = sample_seeds(n_seeds, seed=seed)
    problems = generate_problems(seeds, writer)
    print(f"[gen] {len(problems)}/{n_seeds} parsed")

    problems = F.by_rouge_l(problems, field="question", threshold=0.6)
    print(f"[dedup] {len(problems)}")

    accepted = verify(problems, lite, strong)
    t1 = sum(1 for p in accepted if p["tier"] == 1)
    t2 = sum(1 for p in accepted if p["tier"] == 2)
    print(f"[verify] accepted {len(accepted)}/{len(problems)} (tier1={t1} tier2={t2})")

    cost = (writer.last_usage["total_cost"] + lite.last_usage["total_cost"]
            + strong.last_usage["total_cost"])
    print(f"[cost] uncached this run: ${cost:.4f}  (~${cost/max(1,len(accepted)):.4f}/accepted)")

    out_path = os.path.join(os.path.dirname(__file__), "..", "hard_math", "inspect_v3.jsonl")
    with open(out_path, "w") as f:
        for p in accepted:
            ref = ""
            for tag in ("flash1", "flash2", "lite1", "lite2", "lite3"):
                s = p.get("_sol_" + tag, "")
                if extract_final_int(s) == p["gold"]:
                    ref = s
                    break
            f.write(json.dumps({
                "topic": p["topic"], "subtopic": p["subtopic"], "tier": p["tier"],
                "question": p["question"], "gold": p["gold"],
                "writer_answer": p["writer_answer"], "lite_votes": p["_lite_votes"],
                "reference_solution": ref,
            }) + "\n")
    print(f"[write] {len(accepted)} -> {out_path}")


if __name__ == "__main__":
    main()
