"""Independent validator for the coding_questions dataset.

For every example, reconstruct the full program (prompt + canonical_solution +
test) exactly as a downstream HumanEval-style harness would, run check(candidate)
and the docstring doctests in a subprocess with a timeout, and report pass rate
plus simple diversity stats. This is intentionally separate from the generation
pipeline so it catches any bug in the pipeline's own verifier.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(__file__)
PATH = os.path.join(HERE, "coding_questions.jsonl")

RUNNER = r"""
import sys, json, doctest, io
d = json.loads(sys.stdin.read())
prog = d["prompt"] + "\n" + d["canonical_solution"] + "\n" + d["test"]
ns = {}
exec(prog, ns)
fn = ns[d["entry_point"]]
# doctests
runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
for dt in doctest.DocTestFinder(verbose=False).find(fn, name=d["entry_point"], globs=dict(ns)):
    r = runner.run(dt, out=io.StringIO().write)
    assert r.failed == 0, "doctest failed"
ns["check"](fn)
print("PASS")
"""


def run_one(ex: dict, timeout: float = 10.0) -> tuple[bool, str]:
    try:
        r = subprocess.run([sys.executable, "-c", RUNNER], input=json.dumps(ex),
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    return (r.returncode == 0 and r.stdout.strip().endswith("PASS")), r.stderr.strip()[-200:]


def main() -> None:
    with open(PATH) as f:
        data = [json.loads(line) for line in f if line.strip()]
    npass = 0
    fails = []
    asserts = []
    sol_lens = []
    for ex in data:
        ok, err = run_one(ex)
        npass += ok
        if not ok:
            fails.append((ex.get("entry_point"), err))
        asserts.append(ex["test"].count("assert "))
        sol_lens.append(len(ex["canonical_solution"]))
    print(f"examples={len(data)}  pass={npass}  fail={len(fails)}")
    if data:
        print(f"asserts/problem: min={min(asserts)} avg={sum(asserts)/len(asserts):.1f} max={max(asserts)}")
        print(f"solution chars: min={min(sol_lens)} avg={sum(sol_lens)//len(sol_lens)} max={max(sol_lens)}")
        names = [e.get("entry_point") for e in data]
        print(f"unique entry_points: {len(set(names))}/{len(names)}")
    for ep, err in fails[:15]:
        print(f"  FAIL {ep}: {err[:120]}")


if __name__ == "__main__":
    main()
