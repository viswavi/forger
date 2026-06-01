"""Difficult programming-question dataset — iteration 1 (tiny validation run).

Goal: generate HumanEval-style Python problems, each with:
  - prompt:            function signature + docstring (with >>> doctests)
  - canonical_solution: the function BODY only (indented; appended to prompt)
  - test:              defines ``check(candidate)`` with many asserts

The critical quality gate is EXECUTION: we run ``prompt + canonical_solution``
to define the function, then run ``test`` and call ``check(fn)`` in a sandboxed
subprocess. Only problems whose canonical solution passes their own tests are
kept. Generation uses a delimiter format (not JSON) because code with newlines
and quotes is far more robustly extracted that way.

This iteration generates a handful of problems with BOTH candidate models and
reports the verification pass-rate and cost so we can pick a model for the
full run.
"""
from __future__ import annotations

import os
import re
import sys

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
STRONG_MODEL_ID = "litellm_proxy/gemini/gemini-3.5-flash"


def make_model(model_id: str, max_tokens: int | None = None) -> Model:
    return Model(
        model_id, api_key=API_KEY, api_base=API_BASE,
        temperature=1.0, max_tokens=max_tokens, num_retries=2,
    )


# ---------------------------------------------------------------------------
# Seed grid: algorithmic topics x difficulty twists.
# ---------------------------------------------------------------------------
TOPICS = [
    "dynamic programming",
    "graph traversal (BFS/DFS) on an adjacency structure",
    "shortest paths or weighted graphs",
    "greedy algorithms with a non-obvious exchange argument",
    "string parsing / tokenization",
    "advanced string algorithms (pattern matching, palindromes)",
    "number theory (primes, gcd, modular arithmetic)",
    "combinatorics and counting",
    "backtracking / recursion with pruning",
    "bit manipulation",
    "interval scheduling / merging",
    "matrix / grid simulation",
    "sliding window / two pointers",
    "heaps / priority queues",
    "union-find / disjoint sets",
    "binary search (incl. on the answer)",
    "stack/queue-based simulation",
    "geometry with integer coordinates",
]

TWISTS = [
    "Include a subtle edge case (empty input, single element, ties, overflow-ish values) the tests must cover.",
    "Require careful handling of duplicates or boundary conditions.",
    "Demand an efficient solution (the naive approach would be too slow for large inputs described in the docstring).",
    "Combine two ideas (e.g. sorting + greedy, or hashing + two-pointers).",
    "Use a non-trivial return type (list of tuples, dict, nested structure).",
]


def build_seeds(target: int) -> list[dict]:
    seeds: list[dict] = []
    i = 0
    while len(seeds) < target:
        seeds.append({
            "topic": TOPICS[i % len(TOPICS)],
            "twist": TWISTS[(i // len(TOPICS)) % len(TWISTS)],
            "nonce": i,
        })
        i += 1
    return seeds


GEN_INSTRUCTIONS = """You are an expert competitive-programming problem setter. Design ONE \
challenging, self-contained Python coding problem in the exact style of the HumanEval \
benchmark, but HARDER than a typical HumanEval task.

Topic: {topic}
Design requirement: {twist}

Output EXACTLY these four sections, each wrapped in the shown tags, and NOTHING else \
(no markdown fences, no commentary):

<ENTRY_POINT>the_function_name</ENTRY_POINT>
<PROMPT>
from typing import List          # include only imports the solution needs
<blank line>
<blank line>
def the_function_name(args...) -> ReturnType:
    \"\"\" A clear, precise problem statement describing exactly what to compute,
    including any constraints. Include 2-3 doctest examples using >>> that show
    expected outputs.
    >>> the_function_name(example_input)
    expected_output
    \"\"\"
</PROMPT>
<SOLUTION>
    # The function BODY ONLY, indented by 4 spaces, with NO `def` line.
    # When appended directly after <PROMPT> it must form a complete, correct function.
</SOLUTION>
<TEST>


METADATA = {{
    'author': 'forger',
    'dataset': 'coding_questions'
}}


def check(candidate):
    # 6-12 assert statements covering the examples AND tricky edge cases.
    # Call the function as `candidate(...)`, never by its real name.
    assert candidate(...) == ...
</TEST>

Hard requirements:
- The PROMPT must contain the full `def` line and a docstring; the SOLUTION must be \
ONLY the indented body so that PROMPT + SOLUTION is a valid function.
- The canonical solution MUST be correct and MUST pass every assertion in `check`.
- Tests must reference the function through the `candidate` parameter, not its name.
- Use only the Python standard library. No input()/network/file access.
- Make it genuinely difficult but unambiguous (a single correct output per input)."""


SECTION_RE = {
    "entry_point": re.compile(r"<ENTRY_POINT>\s*(.*?)\s*</ENTRY_POINT>", re.DOTALL),
    "prompt": re.compile(r"<PROMPT>\s*\n(.*?)</PROMPT>", re.DOTALL),
    "solution": re.compile(r"<SOLUTION>\s*\n(.*?)</SOLUTION>", re.DOTALL),
    "test": re.compile(r"<TEST>\s*\n(.*?)</TEST>", re.DOTALL),
}


def parse_problem(text: str) -> dict | None:
    out = {}
    for key, rx in SECTION_RE.items():
        m = rx.search(text)
        if not m:
            return None
        out[key] = m.group(1)
    out["entry_point"] = out["entry_point"].strip()
    # prompt/solution/test keep internal formatting; strip trailing whitespace only.
    out["prompt"] = out["prompt"].rstrip() + "\n"
    out["solution"] = out["solution"].rstrip("\n") + "\n"
    out["test"] = out["test"].rstrip() + "\n"
    if not out["entry_point"] or not re.match(r"^[A-Za-z_]\w*$", out["entry_point"]):
        return None
    if f"def {out['entry_point']}" not in out["prompt"]:
        return None
    return out


def gen_prompt(item: dict) -> str:
    return GEN_INSTRUCTIONS.format(topic=item["topic"], twist=item["twist"])


# ---------------------------------------------------------------------------
# Execution-based verification (run in a subprocess with a timeout).
# ---------------------------------------------------------------------------
VERIFY_SNIPPET = r"""
import sys, json
data = json.loads(sys.stdin.read())
prog = data["prompt"] + "\n" + data["solution"] + "\n" + data["test"]
ns = {}
try:
    exec(prog, ns)
    ns["check"](ns[data["entry_point"]])
except Exception as e:
    import traceback
    print("FAIL:" + repr(e))
    sys.exit(1)
print("PASS")
"""


def verify(problem: dict, timeout: float = 10.0) -> tuple[bool, str]:
    import json
    import subprocess
    payload = json.dumps({
        "prompt": problem["prompt"],
        "solution": problem["solution"],
        "test": problem["test"],
        "entry_point": problem["entry_point"],
    })
    try:
        r = subprocess.run(
            [sys.executable, "-c", VERIFY_SNIPPET],
            input=payload, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    ok = r.returncode == 0 and "PASS" in r.stdout
    return ok, (r.stdout + r.stderr).strip()[:300]


def run_for_model(label: str, model: Model, seeds: list[dict]) -> None:
    raw = generate_from_context(
        seeds, gen_prompt, model,
        samples_per_item=1, output_schema=None,
        temperature=1.0, max_retries=1, batch_size=8,
    )
    parsed = []
    for r in raw:
        if r is None:
            continue
        p = parse_problem(r["text"])
        if p is not None:
            parsed.append(p)
    n_passed = 0
    for p in parsed:
        ok, msg = verify(p)
        n_passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"[{label}] {status} {p['entry_point']}")
        if not ok:
            print("      ", msg.replace("\n", " | ")[:200])
    cost = model.last_usage["total_cost"] if model.last_usage else 0.0
    print(f"\n[{label}] seeds={len(seeds)} parsed={len(parsed)} "
          f"verified={n_passed} last_batch_cost=${cost:.4f}\n")
    if parsed:
        print(f"--- sample problem from [{label}] ---")
        sp = parsed[0]
        print(sp["prompt"])
        print("  # solution:")
        print(sp["solution"])


def main() -> None:
    seeds = build_seeds(6)
    print("=== LITE model ===")
    run_for_model("lite", make_model(LITE_MODEL_ID, max_tokens=2000), seeds)
    print("=== STRONG model ===")
    run_for_model("strong", make_model(STRONG_MODEL_ID, max_tokens=2000), seeds)


if __name__ == "__main__":
    main()
