r"""Check solutions and targets in the sandbox the RL reward uses, many at a time.

``RLSandboxRunner`` runs a solution on tests through ``solution_worker.py``: the
same RestrictedPython sandbox, imports, 4-second budget per run, and resource
limits as ``run_program`` / ``call_method`` inside grade(). It plugs into
Forger's program ops as ``runner=`` and also works on its own::

    from rl.solution_runner import RLSandboxRunner, score_targets
    runner = RLSandboxRunner()
    runner(code, [{"input": "3\\n1 2 3\\n", "output": "6"}])
    filter.by_tests(rows, code_field="solution", runner=runner)

``score_targets`` scores each row's ``ground_truth`` with the row's own code
criteria, exactly as one RL reward would (``rubric_rm.code_score``), so targets
and rubrics that disagree (a solution using ``__slots__``, a grade() that
crashes) show up before training.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .rubric_data import validate_rubric  # noqa: ABS101
from .rubric_rm import _WORKER_ENV, code_score  # noqa: ABS101
from .rubric_worker import RUN_TIMEOUT_S  # noqa: ABS101

_WORKER = Path(__file__).with_name("solution_worker.py")


def _default_workers() -> int:
    return min(16, os.cpu_count() or 1)


class RLSandboxRunner:
    """``(code, tests) -> results`` in the RL reward's solution sandbox.

    A test is ``{"input": stdin, "output": expected stdout}`` for a script, or
    ``{"call": [class_name, method_name, *args], "output": expected}`` for a
    class-based solution, whose return value (JSON-normalized, so tuples become
    lists) must equal ``output``. Each result has ``status`` ("ok",
    "runtime_error", or "timeout"), ``stdout`` (a script's output, or the JSON
    of the returned value), ``stderr`` (the error type, plus the reason when the
    code does not compile in the sandbox, e.g. 'ValueError: sandbox: failed to
    compile code: ... "__slots__" is an invalid variable name ...'),
    ``seconds``, and for calls ``passed``. Scripts are graded by the caller
    (Forger's checker compares stdout with ``output`` up to whitespace). One
    worker process per call; thread-safe, so Forger runs many rows at a time.
    """

    def cache_key(self) -> tuple:
        """Stable key for Forger's op cache: the sandbox's source files."""
        sources = ("solution_worker.py", "rubric_worker.py", "rubric_sandbox.py")
        return ("RLSandboxRunner",) + tuple(
            (Path(__file__).with_name(name)).read_text() for name in sources
        )

    def __repr__(self) -> str:
        """Short name for error messages and reports."""
        return "RLSandboxRunner()"

    def __call__(self, code: str, tests: list[dict]) -> list[dict[str, Any]]:
        """Run ``code`` on every test in one sandboxed worker process."""
        if not tests:
            return []
        payload = json.dumps(
            {
                "code": code,
                "tests": [
                    {"call": list(t["call"])}
                    if "call" in t
                    else {"input": str(t.get("input") or "")}
                    for t in tests
                ],
            }
        )
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="solution-") as directory:
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(_WORKER)],
                    input=payload,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=directory,
                    env=_WORKER_ENV,
                    timeout=(RUN_TIMEOUT_S + 1) * len(tests) + 10,
                )
            except subprocess.TimeoutExpired:
                proc = None
        seconds = (time.monotonic() - start) / len(tests)
        if proc is None or proc.returncode:
            # A crash or kill (e.g. the 512MB limit) fails every test, as
            # rubric_rm scores a crashed criterion 0.
            status = "timeout" if proc is None else "runtime_error"
            error = "worker timed out" if proc is None else proc.stderr[-500:]
            return [
                {"status": status, "stderr": error, "seconds": seconds} for _ in tests
            ]
        results = []
        for test, outcome in zip(tests, json.loads(proc.stdout)["results"]):
            ok, value = outcome["ok"], outcome["value"]
            if not ok:
                status = "timeout" if value == "SandboxTimeout" else "runtime_error"
                if "message" in outcome:
                    value = f"{value}: {outcome['message']}"
                results.append({"status": status, "stderr": value, "seconds": seconds})
                continue
            result: dict[str, Any] = {"status": "ok", "seconds": seconds}
            if "call" in test:
                result["stdout"] = json.dumps(value)
                if test.get("output") is not None:
                    expected = json.loads(json.dumps(test["output"]))
                    result["passed"] = value == expected
            else:
                result["stdout"] = value
            results.append(result)
        return results


def run_solution(code: str, tests: list[dict]) -> list[dict[str, Any]]:
    """Run ``code`` on ``tests`` in the RL sandbox and grade each one.

    Without Forger: scripts pass when stdout equals ``output`` up to
    whitespace, calls when the returned value equals ``output``. Each result
    gets ``passed`` (None for a test without ``output``).
    """
    results = RLSandboxRunner()(code, tests)
    for test, result in zip(tests, results):
        if test.get("output") is None:
            result["passed"] = None
        elif result["status"] != "ok":
            result["passed"] = False
        elif "passed" not in result:
            result["passed"] = result["stdout"].split() == str(test["output"]).split()
    return results


def score_targets(
    rows: list[dict],
    *,
    response_field: str = "ground_truth",
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    """Score each row's target with its own code criteria, as RL would.

    Returns one dict per row: ``score`` (the weighted mean over the code
    criteria; None when the row has none), ``criteria`` (one score per rubric
    entry, None for prompt criteria, which need the judge and are not run),
    and ``prompt_weight`` (the weight left unscored). A correct target whose
    score is below 1 means its rubric or its code does not survive the RL
    sandbox. Rows may be raw submission rows or validated ones.
    """
    jobs = []
    for i, row in enumerate(rows):
        rubric = validate_rubric(row["rubric"])
        prompt = row["prompt"]
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        for j, (criterion, weight) in enumerate(rubric):
            if "code" in criterion and weight:
                jobs.append((i, j, criterion["code"], prompt, row[response_field], row))
    with ThreadPoolExecutor(max_workers=max_workers or _default_workers()) as pool:
        values = list(
            pool.map(
                lambda job: code_score(job[2], job[3], job[4], job[5]["ground_truth"]),
                jobs,
            )
        )
    scores: list[list[float | None]] = [[None] * len(row["rubric"]) for row in rows]
    for (i, j, *_), value in zip(jobs, values):
        scores[i][j] = value
    out = []
    for row, criteria in zip(rows, scores):
        weights = [float(weight) for _, weight in row["rubric"]]
        code_weight = sum(w for w, s in zip(weights, criteria) if s is not None)
        out.append(
            {
                "score": (
                    sum(w * s for w, s in zip(weights, criteria) if s is not None)
                    / code_weight
                    if code_weight
                    else None
                ),
                "criteria": criteria,
                "prompt_weight": sum(weights) - code_weight,
            }
        )
    return out
