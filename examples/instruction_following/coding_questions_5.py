"""Difficult programming-question dataset — iteration 5 (calibrated quality scoring).

Iteration 4's judge at score_threshold=5 was far too strict (62/426). Instead of
guessing a threshold, this pass SCORES every example 1-5 with a single judge call
(cached via generate_from_context), prints the score distribution, and writes the
survivors at a chosen cutoff. Scores are kept in coding_questions_scored.jsonl so
the cutoff can be changed without re-calling the model.

Gates applied first (cheap, deterministic):
  * input-usage: solution must reference >=1 parameter (drops hardcoded stubs).
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("LITELLM_API_KEY", "sk-n35XeqvGumG4V_Z3wMQAjA")
os.environ.setdefault(
    "FORGER_CACHE", os.path.join(os.path.dirname(__file__), ".forger_cache")
)

from pyforger.acquire import Model  # noqa: E402
from pyforger.alter.generate import generate_from_context  # noqa: E402

API_KEY = os.environ["LITELLM_API_KEY"]
API_BASE = "https://cmu.litellm.ai"
LITE_MODEL_ID = "litellm_proxy/gemini/gemini-3.1-flash-lite"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "coding_questions")
RAW = os.path.join(DATA_DIR, "coding_questions_raw.jsonl")
SCORED = os.path.join(DATA_DIR, "coding_questions_scored.jsonl")
OUT = os.path.join(DATA_DIR, "coding_questions.jsonl")

CUTOFF = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def make_model() -> Model:
    return Model(LITE_MODEL_ID, api_key=API_KEY, api_base=API_BASE,
                 temperature=1.0, max_tokens=400, num_retries=2)


def param_names(prompt: str, entry_point: str) -> list[str]:
    for closer in ("\n    pass\n", '\n    """\n    pass\n'):
        try:
            tree = ast.parse(prompt + closer)
            break
        except SyntaxError:
            tree = None
    if tree is None:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            a = node.args
            names = [p.arg for p in (a.posonlyargs + a.args + a.kwonlyargs)]
            return [n for n in names if n != "self"]
    return []


def uses_inputs(ex: dict) -> bool:
    params = param_names(ex["prompt"], ex["entry_point"])
    if not params:
        return True
    body = ex["canonical_solution"]
    return any(re.search(rf"\b{re.escape(p)}\b", body) for p in params)


def score_prompt(ex: dict) -> str:
    return (
        "You are grading a candidate problem for a training set of HARD Python "
        "problems. Rate it 1-5 on this rubric:\n"
        "5 = solution is a correct, general algorithm for the stated problem and "
        "the tests meaningfully exercise it;\n"
        "4 = solution is essentially correct/general with at most a minor nit;\n"
        "3 = mostly reasonable but some doubt about correctness or test coverage;\n"
        "2 = solution looks hardcoded/overfit, mismatches the problem, or is likely "
        "wrong for some valid input;\n"
        "1 = clearly wrong, trivial, or a stub.\n\n"
        f"PROBLEM STATEMENT:\n{ex['prompt']}\n\n"
        f"CANONICAL SOLUTION:\n{ex['canonical_solution']}\n\n"
        f"TESTS:\n{ex['test']}\n\n"
        'Respond with JSON only: {"score": <int 1-5>, "reason": "<short>"}'
    )


def main() -> None:
    with open(RAW) as f:
        data = [json.loads(line) for line in f if line.strip()]
    n0 = len(data)
    data = [ex for ex in data if uses_inputs(ex)]
    n1 = len(data)

    model = make_model()
    scored = generate_from_context(
        data, score_prompt, model,
        samples_per_item=1, output_schema={"score": int, "reason": str},
        temperature=0.0, max_retries=2, batch_size=16,
    )

    for ex, s in zip(data, scored):
        if s is None:
            ex["_score"] = None
        else:
            try:
                ex["_score"] = int(s.get("score"))
            except (TypeError, ValueError):
                ex["_score"] = None

    dist = Counter(ex["_score"] for ex in data)
    print(f"raw={n0}  after input-usage gate={n1}")
    print("score distribution:", dict(sorted(dist.items(), key=lambda kv: (kv[0] is None, kv[0]))))
    for c in (3, 4, 5):
        print(f"  >= {c}: {sum(1 for ex in data if ex['_score'] is not None and ex['_score'] >= c)}")

    # persist scores
    with open(SCORED, "w") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    kept = [ex for ex in data if ex["_score"] is not None and ex["_score"] >= CUTOFF]
    for ex in kept:
        ex.pop("_score", None)
    with open(OUT, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    cost = model.last_usage["total_cost"] if model.last_usage else 0.0
    print(f"cutoff>={CUTOFF}: wrote {len(kept)} -> {OUT}  (judge last-batch ${cost:.4f})")


if __name__ == "__main__":
    main()
