"""Difficult programming-question dataset — iteration 2.

Iteration-1 findings:
  * The LITE model (gemini-3.1-flash-lite) follows the delimiter format reliably
    and is cheap (~$0.0014 / problem). The STRONG model (gemini-3.5-flash) is a
    verbose "thinking" model that spends its tokens on reasoning, ignores the
    output format, and costs ~13x more. -> use LITE.
  * First-pass execution yield was only ~33%: the model often writes a problem
    whose canonical solution and tests disagree.

This iteration keeps the LITE model but adds:
  1. EXECUTION verification = run prompt+solution+test, call check(fn), AND run the
     docstring doctests, all in a sandboxed subprocess with a timeout. A problem is
     "valid" only if every assert and every doctest passes (self-consistent).
  2. SELF-REPAIR: failed problems get one repair round where the model sees the
     failure traceback and returns a corrected problem; we re-verify.
  3. Quality gates: non-trivial solution, >= 5 asserts, std-lib only.
  4. ROUGE-L dedup on the prompt + unique entry-point names.

Set TEST=True for a small run that prints stats; TEST=False writes JSONL.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("LITELLM_API_KEY", "sk-n35XeqvGumG4V_Z3wMQAjA")
os.environ.setdefault(
    "FORGER_CACHE", os.path.join(os.path.dirname(__file__), ".forger_cache")
)

import pyforger  # noqa: E402
from pyforger import filter as ffilter  # noqa: E402
from pyforger import op  # noqa: E402
from pyforger.acquire import Model  # noqa: E402
from pyforger.alter.generate import generate_from_context  # noqa: E402

API_KEY = os.environ["LITELLM_API_KEY"]
API_BASE = "https://cmu.litellm.ai"
LITE_MODEL_ID = "litellm_proxy/gemini/gemini-3.1-flash-lite"

TEST = True
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "coding_questions")


def make_model(model_id: str, max_tokens: int | None = None) -> Model:
    return Model(
        model_id, api_key=API_KEY, api_base=API_BASE,
        temperature=1.0, max_tokens=max_tokens, num_retries=2,
    )


# ---------------------------------------------------------------------------
# Seed grid: algorithmic topics x difficulty twists.
# ---------------------------------------------------------------------------
TOPICS = [
    "dynamic programming over sequences",
    "dynamic programming on grids",
    "graph traversal (BFS/DFS) on an adjacency structure",
    "shortest paths or weighted graphs",
    "topological sorting / dependency resolution",
    "greedy algorithms with a non-obvious choice",
    "string parsing / tokenization of a small grammar",
    "pattern matching and palindromes",
    "run-length / encoding-decoding transformations",
    "number theory (primes, gcd, modular arithmetic)",
    "combinatorics and counting",
    "backtracking / recursion with pruning",
    "bit manipulation and bitmask tricks",
    "interval scheduling / merging / overlap",
    "matrix / grid simulation and transformations",
    "sliding window / two pointers",
    "heaps / priority queues / k-th element",
    "union-find / connected components",
    "binary search, including binary search on the answer",
    "stack/queue-based simulation (e.g. expression evaluation)",
    "geometry with integer coordinates",
    "hashing and frequency analysis",
    "sorting with custom comparators / stable tie-breaking",
    "tree problems on a parent/children structure",
    "state-machine / rules simulation",
]

TWISTS = [
    "Include a subtle edge case (empty input, single element, ties, negatives) the tests must cover.",
    "Require careful handling of duplicates or boundary conditions.",
    "Combine two ideas (e.g. sorting + greedy, or hashing + two-pointers).",
    "Use a non-trivial return type (list of tuples, dict, or nested structure).",
    "Require returning ALL optimal solutions / a canonicalized result, not just a count.",
    "Add a constraint that rules out the most obvious naive approach.",
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


FORMAT_SPEC = """Output EXACTLY these four sections, each wrapped in the shown tags, and NOTHING \
else (no markdown fences, no commentary before or after):

<ENTRY_POINT>the_function_name</ENTRY_POINT>
<PROMPT>
from typing import List          # include ONLY imports the solution needs
<blank line>
<blank line>
def the_function_name(args...) -> ReturnType:
    \"\"\" A precise problem statement describing exactly what to compute, with any
    constraints. Then 2-3 doctest examples using >>> whose printed results EXACTLY
    match what the canonical solution returns.
    >>> the_function_name(example_input)
    expected_output
    \"\"\"
</PROMPT>
<SOLUTION>
    # The function BODY ONLY, indented by 4 spaces, NO `def` line. PROMPT + SOLUTION
    # together must form one complete, correct function.
</SOLUTION>
<TEST>


METADATA = {{
    'author': 'forger',
    'dataset': 'coding_questions'
}}


def check(candidate):
    # At least 6 assert statements covering the doctest examples AND tricky edge
    # cases. Always call the function as `candidate(...)`, never by its real name.
    assert candidate(...) == ...
</TEST>

Iron rules (a violation makes the problem useless):
- PROMPT contains the full `def` line + docstring; SOLUTION is ONLY the indented body.
- The canonical solution MUST be correct: mentally execute it on every doctest and
  every assert and confirm they all pass. The doctests in the PROMPT must match the
  solution's real output EXACTLY.
- Tests reference the function only through the `candidate` parameter.
- Use ONLY the Python standard library. No input()/network/file/random access.
- Deterministic: every input has exactly one correct output."""


GEN_INSTRUCTIONS = (
    "You are an expert competitive-programming problem setter. Design ONE "
    "challenging, self-contained Python coding problem in the exact style of the "
    "HumanEval benchmark, but clearly HARDER than a typical HumanEval task (it should "
    "require a real algorithmic insight, not just a loop).\n\n"
    "Topic: {topic}\n"
    "Design requirement: {twist}\n\n"
) + FORMAT_SPEC


def gen_prompt(item: dict) -> str:
    return GEN_INSTRUCTIONS.format(topic=item["topic"], twist=item["twist"])


def repair_prompt(item: dict) -> str:
    return (
        "The following HumanEval-style Python problem is BROKEN: running its canonical "
        "solution against its own tests (or doctests) fails. Here is the failure:\n\n"
        f"---ERROR---\n{item['_error']}\n---END ERROR---\n\n"
        "Here is the broken problem:\n\n"
        f"<ENTRY_POINT>{item['entry_point']}</ENTRY_POINT>\n"
        f"<PROMPT>\n{item['prompt']}</PROMPT>\n"
        f"<SOLUTION>\n{item['solution']}</SOLUTION>\n"
        f"<TEST>\n{item['test']}</TEST>\n\n"
        "Fix it so the canonical solution is CORRECT and passes every doctest and every "
        "assert. You may fix the solution, the tests, or the doctests, but keep the same "
        "general problem idea and entry-point name. Re-derive the expected outputs "
        "carefully by mentally executing the solution.\n\n" + FORMAT_SPEC
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
SECTION_RE = {
    "entry_point": re.compile(r"<ENTRY_POINT>\s*(.*?)\s*</ENTRY_POINT>", re.DOTALL),
    "prompt": re.compile(r"<PROMPT>\s*\n(.*?)</PROMPT>", re.DOTALL),
    "solution": re.compile(r"<SOLUTION>\s*\n(.*?)</SOLUTION>", re.DOTALL),
    "test": re.compile(r"<TEST>\s*\n(.*?)</TEST>", re.DOTALL),
}


def parse_problem(text: str) -> dict | None:
    out: dict = {}
    for key, rx in SECTION_RE.items():
        m = rx.search(text)
        if not m:
            return None
        out[key] = m.group(1)
    out["entry_point"] = out["entry_point"].strip()
    out["prompt"] = out["prompt"].rstrip() + "\n"
    out["solution"] = out["solution"].rstrip("\n") + "\n"
    out["test"] = out["test"].rstrip() + "\n"
    if not re.match(r"^[A-Za-z_]\w*$", out["entry_point"]):
        return None
    if f"def {out['entry_point']}" not in out["prompt"]:
        return None
    return out


def quality_ok(p: dict) -> bool:
    body = p["solution"].strip()
    if len(body) < 30:                      # too trivial
        return False
    if p["test"].count("assert ") < 5:      # too few tests
        return False
    if "candidate" not in p["test"]:
        return False
    banned = ("input(", "open(", "import os", "import sys", "import socket",
              "import requests", "subprocess", "__import__", "eval(", "exec(")
    blob = p["solution"] + p["test"]
    return not any(b in blob for b in banned)


# ---------------------------------------------------------------------------
# Execution verification: asserts + doctests, sandboxed subprocess + timeout.
# ---------------------------------------------------------------------------
VERIFY_SNIPPET = r"""
import sys, json, doctest
data = json.loads(sys.stdin.read())
prompt, solution, test, ep = (
    data["prompt"], data["solution"], data["test"], data["entry_point"])
ns = {}
try:
    exec(prompt + "\n" + solution, ns)
except Exception as e:
    import traceback; print("DEFINE_FAIL: " + traceback.format_exc()); sys.exit(1)
fn = ns.get(ep)
if fn is None:
    print("NO_ENTRY: " + ep); sys.exit(1)
# Run the docstring doctests.
runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
finder = doctest.DocTestFinder(verbose=False)
try:
    found = finder.find(fn, name=ep, globs=dict(ns))
except Exception as e:
    import traceback; print("DOCTEST_FIND_FAIL: " + traceback.format_exc()); sys.exit(1)
import io
buf = io.StringIO()
total_examples = 0
for dt in found:
    total_examples += len(dt.examples)
    res = runner.run(dt, out=buf.write)
    if res.failed:
        print("DOCTEST_FAIL:\n" + buf.getvalue()[:1500]); sys.exit(1)
# Run the assert-based check().
try:
    exec(test, ns)
    ns["check"](fn)
except Exception as e:
    import traceback; print("CHECK_FAIL: " + traceback.format_exc()[-1500:]); sys.exit(1)
print("PASS doctests=%d" % total_examples)
"""


def verify(problem: dict, timeout: float = 8.0) -> tuple[bool, str]:
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
    ok = r.returncode == 0 and r.stdout.startswith("PASS")
    return ok, (r.stdout + r.stderr).strip()[:1500]


# ---------------------------------------------------------------------------
# Pipeline steps (each @op-cached on disk).
# ---------------------------------------------------------------------------
@op(name="cq.generate")
def generate(seeds, model):
    raw = generate_from_context(
        seeds, gen_prompt, model,
        samples_per_item=1, output_schema=None,
        temperature=1.0, max_retries=1, batch_size=16,
    )
    parsed = []
    for r in raw:
        if r is None:
            continue
        p = parse_problem(r["text"])
        if p is not None and quality_ok(p):
            parsed.append(p)
    return parsed


@op(name="cq.repair")
def repair(broken, model):
    """One repair round for problems that failed verification."""
    if not broken:
        return []
    raw = generate_from_context(
        broken, repair_prompt, model,
        samples_per_item=1, output_schema=None,
        temperature=1.0, max_retries=1, batch_size=16,
    )
    fixed = []
    for r in raw:
        if r is None:
            continue
        p = parse_problem(r["text"])
        if p is not None and quality_ok(p):
            fixed.append(p)
    return fixed


def verify_split(problems):
    good, bad = [], []
    for p in problems:
        ok, msg = verify(p)
        if ok:
            good.append(p)
        else:
            p = dict(p)
            p["_error"] = msg
            bad.append(p)
    return good, bad


def build_dataset(seeds, model):
    problems = generate(seeds, model)
    good, bad = verify_split(problems)
    # one repair round
    repaired = repair(bad, model)
    good2, _ = verify_split(repaired)
    good.extend(good2)
    # dedup: unique entry points then ROUGE-L on prompt
    seen, uniq = set(), []
    for p in good:
        if p["entry_point"] in seen:
            continue
        seen.add(p["entry_point"])
        uniq.append(p)
    uniq = ffilter.by_rouge_l(uniq, field="prompt", threshold=0.6)
    return problems, good, uniq


def main() -> None:
    model = make_model(LITE_MODEL_ID, max_tokens=2200)
    target = 16 if TEST else 600
    seeds = build_seeds(target)

    problems, good, final = pyforger.run(build_dataset, seeds, model)
    cost = model.last_usage["total_cost"] if model.last_usage else 0.0

    print(f"\nseeds={len(seeds)}  parsed&quality_ok={len(problems)}  "
          f"verified(incl. repair)={len(good)}  final(dedup)={len(final)}")
    print(f"(last batch cost ${cost:.4f})")

    if TEST:
        for p in final[:3]:
            print("=" * 80)
            print(p["prompt"], end="")
            print("  # --- canonical_solution ---")
            print(p["solution"], end="")
    else:
        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(OUT_DIR, "coding_questions.jsonl")
        with open(out, "w") as f:
            for p in final:
                f.write(json.dumps({
                    "prompt": p["prompt"],
                    "canonical_solution": p["solution"],
                    "test": p["test"],
                    "entry_point": p["entry_point"],
                }, ensure_ascii=False) + "\n")
        print(f"Wrote {len(final)} problems to {out}")


if __name__ == "__main__":
    main()
