"""Difficult programming-question dataset — iteration 4 (quality filtering).

Execution-consistency (iteration 3) guarantees the canonical solution agrees
with its own tests, but NOT that the solution genuinely solves the stated
natural-language problem. Spot checks found a few "test-overfit" cases, e.g. a
solution that just returns a hardcoded constant ignoring its inputs, passing
because the auto-generated tests were also weak.

This pass reads coding_questions_raw.jsonl and applies two quality gates,
writing the survivors back to coding_questions.jsonl:

  1. Heuristic: the solution body must reference at least one of the function's
     parameters (rejects hardcoded constant-return "solutions").
  2. LLM judge (by_llm_score, single strict judge at temperature 0): score the
     (problem, solution+tests) triple on whether the solution is a correct,
     GENERAL algorithm for the stated problem and the tests meaningfully
     exercise it. Keep only score == max.
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("LITELLM_API_KEY", "sk-n35XeqvGumG4V_Z3wMQAjA")
os.environ.setdefault(
    "FORGER_CACHE", os.path.join(os.path.dirname(__file__), ".forger_cache")
)

import pyforger  # noqa: E402
from pyforger import filter as ffilter  # noqa: E402
from pyforger.acquire import Model  # noqa: E402

API_KEY = os.environ["LITELLM_API_KEY"]
API_BASE = "https://cmu.litellm.ai"
LITE_MODEL_ID = "litellm_proxy/gemini/gemini-3.1-flash-lite"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "coding_questions")
RAW = os.path.join(DATA_DIR, "coding_questions_raw.jsonl")
OUT = os.path.join(DATA_DIR, "coding_questions.jsonl")


def make_model() -> Model:
    return Model(LITE_MODEL_ID, api_key=API_KEY, api_base=API_BASE,
                 temperature=1.0, max_tokens=200, num_retries=2)


def param_names(prompt: str, entry_point: str) -> list[str]:
    """Extract parameter names of the entry-point function from the prompt."""
    try:
        tree = ast.parse(prompt + "\n    pass\n")
    except SyntaxError:
        # prompt ends mid-docstring; close it crudely for parsing
        try:
            tree = ast.parse(prompt + '\n    """\n    pass\n')
        except SyntaxError:
            return []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            a = node.args
            names = [p.arg for p in (a.posonlyargs + a.args + a.kwonlyargs)]
            if a.vararg:
                names.append(a.vararg.arg)
            if a.kwarg:
                names.append(a.kwarg.arg)
            return [n for n in names if n != "self"]
    return []


def uses_inputs(ex: dict) -> bool:
    """Reject solutions that ignore every parameter (likely hardcoded)."""
    params = param_names(ex["prompt"], ex["entry_point"])
    if not params:                      # nullary function: nothing to check
        return True
    body = ex["canonical_solution"]
    return any(re.search(rf"\b{re.escape(p)}\b", body) for p in params)


JUDGE_CRITERIA = (
    "You are grading a candidate programming problem for a training set of HARD "
    "problems. The text under review combines a CANONICAL SOLUTION and its TEST "
    "suite for the problem stated above. Give the MAX score only if ALL hold: "
    "(a) the solution is a CORRECT, GENERAL algorithm that solves the stated "
    "problem for arbitrary valid inputs (NOT hardcoded to specific test cases, "
    "NOT a stub that ignores its inputs, NOT obviously wrong for some valid "
    "input); (b) the solution genuinely matches what the problem statement asks "
    "for (not a trivially different/easier task); (c) the tests are non-trivial "
    "and actually exercise the core logic, including at least one edge case. "
    "Give a LOW score if the solution looks hardcoded/overfit to the tests, is "
    "incorrect in general, mismatches the problem, or the tests are trivial."
)


def main() -> None:
    with open(RAW) as f:
        data = [json.loads(line) for line in f if line.strip()]
    n0 = len(data)

    # 1. heuristic input-usage gate
    data = [ex for ex in data if uses_inputs(ex)]
    n1 = len(data)

    # add a combined field for the judge to inspect solution AND tests together
    for ex in data:
        ex["_review"] = (
            "CANONICAL SOLUTION:\n" + ex["canonical_solution"]
            + "\n\nTESTS:\n" + ex["test"]
        )

    model = make_model()
    kept = ffilter.by_llm_score(
        data, model=model,
        input_field="prompt", output_field="_review",
        score_threshold=5, max_score=5, num_judges=1,
        temperature=0.0, criteria=JUDGE_CRITERIA,
    )
    n2 = len(kept)

    for ex in kept:
        ex.pop("_review", None)

    with open(OUT, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    cost = model.last_usage["total_cost"] if model.last_usage else 0.0
    print(f"raw={n0}  after input-usage gate={n1}  after LLM judge={n2}")
    print(f"(judge last-batch cost ${cost:.4f}; wrote {n2} -> {OUT})")


if __name__ == "__main__":
    main()
