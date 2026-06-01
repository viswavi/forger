"""Difficult programming-question dataset — iteration 3 (production run).

Same pipeline as iteration 2 (LITE model + execution/doctest verification +
one self-repair round + dedup), with two additions for the real run:
  * CostModel: accumulates total $ spend across ALL calls (Model.last_usage only
    keeps the most recent batch), so we can stay under budget.
  * CLI: `python coding_questions_3.py <n_seeds> [test]`. With `test` it prints
    stats only; otherwise it appends verified problems to the JSONL output.

Run repeatedly with different seed offsets to grow the dataset; each problem is
keyed by a unique entry-point name across the whole file so reruns dedup safely.
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

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "coding_questions")


class CostModel(Model):
    """Model that accumulates total cost across every generate call."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.cumulative_cost = 0.0

    def generate_batch(self, *a, **kw):
        out = super().generate_batch(*a, **kw)
        if self._last_usage:
            self.cumulative_cost += self._last_usage.get("total_cost", 0.0) or 0.0
        return out


def make_model(model_id: str, max_tokens: int | None = None) -> CostModel:
    return CostModel(
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
    "prefix sums / difference arrays",
    "monotonic stack / next-greater-element",
    "game theory / optimal play on small states",
    "string transformations with edit operations",
    "scheduling with deadlines and penalties",
]

TWISTS = [
    "Include a subtle edge case (empty input, single element, ties, negatives) the tests must cover.",
    "Require careful handling of duplicates or boundary conditions.",
    "Combine two ideas (e.g. sorting + greedy, or hashing + two-pointers).",
    "Use a non-trivial return type (list of tuples, dict, or nested structure).",
    "Require returning ALL optimal solutions / a canonicalized result, not just a count.",
    "Add a constraint that rules out the most obvious naive approach.",
    "Require a tie-breaking rule (e.g. lexicographically smallest) that the tests check.",
]


def build_seeds(target: int, offset: int = 0) -> list[dict]:
    seeds: list[dict] = []
    i = offset
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
# Parsing + quality gates
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
    if len(body) < 30:
        return False
    if p["test"].count("assert ") < 5:
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
import sys, json, doctest, io
data = json.loads(sys.stdin.read())
prompt, solution, test, ep = (
    data["prompt"], data["solution"], data["test"], data["entry_point"])
ns = {}
try:
    exec(prompt + "\n" + solution, ns)
except Exception:
    import traceback; print("DEFINE_FAIL: " + traceback.format_exc()); sys.exit(1)
fn = ns.get(ep)
if fn is None:
    print("NO_ENTRY: " + ep); sys.exit(1)
runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
finder = doctest.DocTestFinder(verbose=False)
try:
    found = finder.find(fn, name=ep, globs=dict(ns))
except Exception:
    import traceback; print("DOCTEST_FIND_FAIL: " + traceback.format_exc()); sys.exit(1)
buf = io.StringIO()
total_examples = 0
for dt in found:
    total_examples += len(dt.examples)
    res = runner.run(dt, out=buf.write)
    if res.failed:
        print("DOCTEST_FAIL:\n" + buf.getvalue()[:1500]); sys.exit(1)
try:
    exec(test, ns)
    ns["check"](fn)
except Exception:
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
# Pipeline steps
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
    repaired = repair(bad, model)
    good2, _ = verify_split(repaired)
    good.extend(good2)
    return problems, good


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    is_test = (len(sys.argv) > 3 and sys.argv[3] == "test")

    model = make_model(LITE_MODEL_ID, max_tokens=2200)
    seeds = build_seeds(n, offset=offset)

    problems, good = pyforger.run(build_dataset, seeds, model)

    print(f"\nseeds={len(seeds)} offset={offset}  parsed&quality_ok={len(problems)}  "
          f"verified(incl. repair)={len(good)}")
    print(f"TOTAL COST this run: ${model.cumulative_cost:.4f}")

    if is_test:
        for p in good[:2]:
            print("=" * 80)
            print(p["prompt"], end="")
            print("  # --- canonical_solution ---")
            print(p["solution"], end="")
            print("  # --- test ---")
            print(p["test"][:400])
        return

    # Append to JSONL, deduping against whatever is already on disk by entry_point
    # and ROUGE-L on the prompt.
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "coding_questions.jsonl")
    existing = []
    if os.path.exists(out):
        with open(out) as f:
            existing = [json.loads(line) for line in f if line.strip()]
    # Dedup only on near-duplicate problem TEXT, not on function-name collisions:
    # two different problems that happen to share an entry-point name are both valid.
    seen_prompts = {e.get("prompt") for e in existing}

    merged = list(existing)
    added = 0
    for p in good:
        if p["prompt"] in seen_prompts:
            continue
        seen_prompts.add(p["prompt"])
        merged.append({
            "prompt": p["prompt"],
            "canonical_solution": p["solution"],
            "test": p["test"],
            "entry_point": p["entry_point"],
        })
        added += 1

    merged = ffilter.by_rouge_l(merged, field="prompt", threshold=0.7)

    with open(out, "w") as f:
        for e in merged:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Added {added} new; after dedup total={len(merged)} -> {out}")


if __name__ == "__main__":
    main()
