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
crashes) show up before training. It also scores control responses that should
earn nothing (an abstention on a solvable row, no answer, an empty program), so
a rubric that rewards them shows up too. ``rubric_verifier`` scores a sampled response
the same way, e.g. to measure with ``filter.by_pass_at_k`` how often a
checkpoint already earns full reward on a row.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
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


def _code_jobs(row: dict, response: str) -> list[tuple[int, str, list, str, str]]:
    """The row's weighted code criteria as (index, code, prompt, response, target)."""
    rubric = validate_rubric(row["rubric"])
    prompt = row["prompt"]
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    return [
        (j, criterion["code"], prompt, response, row["ground_truth"])
        for j, (criterion, weight) in enumerate(rubric)
        if "code" in criterion and weight
    ]


def _weighted(row: dict, criteria: list[float | None]) -> dict[str, Any]:
    """The weighted mean over the scored criteria, as score_targets reports it."""
    weights = [float(weight) for _, weight in row["rubric"]]
    code_weight = sum(w for w, s in zip(weights, criteria) if s is not None)
    return {
        "score": (
            sum(w * s for w, s in zip(weights, criteria) if s is not None) / code_weight
            if code_weight
            else None
        ),
        "criteria": criteria,
        "prompt_weight": sum(weights) - code_weight,
    }


def _abstention_marker() -> str:
    """prompt_building.ABSTENTION_MARKER, read without importing LiveCodeBench."""
    source = Path(__file__).resolve().parents[1] / "prompt_building.py"
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and [
            getattr(t, "id", None) for t in node.targets
        ] == ["ABSTENTION_MARKER"]:
            return ast.literal_eval(node.value)
    raise RuntimeError(f"no ABSTENTION_MARKER in {source}")


def control_responses() -> dict[str, str]:
    """Responses that should earn (almost) nothing on a solvable row.

    ``abstain`` is the abstention answer (the right answer on an impossible
    row, so score_targets skips it on rows whose target abstains),
    ``no_answer`` is thinking with nothing after it, and ``empty_program`` is
    a python fence holding ``pass``.
    """
    thinking = "<think>\nI have thought about this task.\n</think>\n\n"
    return {
        "abstain": thinking + _abstention_marker(),
        "no_answer": thinking,
        "empty_program": thinking + "```python\npass\n```",
    }


def _scores(
    rows: list[dict], responses: list[str], max_workers: int | None
) -> list[dict[str, Any]]:
    """Score ``responses[i]`` with ``rows[i]``'s code criteria, in parallel."""
    jobs = [
        (i, *job)
        for i, (row, response) in enumerate(zip(rows, responses))
        for job in _code_jobs(row, response)
    ]
    with ThreadPoolExecutor(max_workers=max_workers or _default_workers()) as pool:
        values = list(pool.map(lambda job: code_score(*job[2:]), jobs))
    scores: list[list[float | None]] = [[None] * len(row["rubric"]) for row in rows]
    for (i, j, *_), value in zip(jobs, values):
        scores[i][j] = value
    return [_weighted(row, criteria) for row, criteria in zip(rows, scores)]


def score_targets(
    rows: list[dict],
    *,
    response_field: str = "ground_truth",
    controls: bool = True,
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    """Score each row's target with its own code criteria, as RL would.

    Returns one dict per row: ``score`` (the weighted mean over the code
    criteria; None when the row has none), ``criteria`` (one score per rubric
    entry, None for prompt criteria, which need the judge and are not run),
    and ``prompt_weight`` (the weight left unscored). A correct target whose
    score is below 1 means its rubric or its code does not survive the RL
    sandbox. Rows may be raw submission rows or validated ones.

    With ``controls``, each dict also gets ``controls``: the score of every
    ``control_responses()`` entry on that row (no ``abstain`` when the target
    itself abstains). RL rewards a control the rubric scores highly, e.g. a
    solvable row's rubric that gives the abstention 1.0 teaches the policy to
    abstain. A warning counts the rows where a control earns at least half
    of the target's score.
    """
    results = _scores(rows, [row[response_field] for row in rows], max_workers)
    if not controls:
        return results
    marker = _abstention_marker()
    flagged: dict[str, list[int]] = {}
    for name, response in control_responses().items():
        indices = [
            i
            for i, row in enumerate(rows)
            if name != "abstain" or not row[response_field].rstrip().endswith(marker)
        ]
        picked = [rows[i] for i in indices]
        for i, control in zip(
            indices, _scores(picked, [response] * len(picked), max_workers)
        ):
            results[i].setdefault("controls", {})[name] = control["score"]
            target = results[i]["score"]
            if (
                control["score"]
                and target is not None
                and control["score"] >= target / 2
            ):
                flagged.setdefault(name, []).append(i)
    for result in results:
        result.setdefault("controls", {})
    if flagged:
        rows_hit = sorted({i for indices in flagged.values() for i in indices})
        counts = ", ".join(
            f"{name}: {len(indices)}" for name, indices in flagged.items()
        )
        warnings.warn(
            f"score_targets: in {len(rows_hit)} of {len(rows)} rows a control response "
            f"earns at least half of the target's score ({counts}), so RL rewards it "
            f"too; e.g. rows {rows_hit[:10]}. Each row's 'controls' has the scores.",
            stacklevel=2,
        )
    return results


class RubricVerifier:
    """``(response, row) -> bool | None``: see ``rubric_verifier``."""

    def __init__(self, threshold: float = 1.0) -> None:
        """Pass a response whose weighted code score is at least ``threshold``."""
        self.threshold = threshold

    def cache_key(self) -> tuple:
        """Stable key for Forger's op cache: the threshold and the grader's sources."""
        sources = ("rubric_rm.py", "rubric_worker.py", "rubric_sandbox.py")
        return ("RubricVerifier", self.threshold) + tuple(
            (Path(__file__).with_name(name)).read_text() for name in sources
        )

    def __repr__(self) -> str:
        """Short name for error messages and reports."""
        return f"rubric_verifier({self.threshold})"

    def __call__(self, response: str, row: dict) -> bool | None:
        """Score ``response`` with the row's code criteria in the RL sandbox."""
        try:
            jobs = _code_jobs(row, response)
            criteria: list[float | None] = [None] * len(row["rubric"])
            for j, *job in jobs:
                criteria[j] = code_score(*job)
        except Exception:
            return None
        score = _weighted(row, criteria)["score"]
        return None if score is None else score >= self.threshold


def rubric_verifier(threshold: float = 1.0) -> RubricVerifier:
    """Forger verifier: does a response score at least ``threshold`` on its rubric?

    ``verify(response, row)`` scores ``response`` with the row's code criteria
    as ``score_targets`` scores a target: the RL reward's sandbox, timeouts and
    weights, averaged over the code criteria (prompt criteria need the judge
    and are not run). None when the row has no weighted code criteria or
    grading fails (an invalid rubric, a sandbox that cannot start).

    RL grades the whole response, thinking included, so give it the whole
    completion: with ``filter.by_pass_at_k``, pass ``extract_answer=lambda c:
    c`` (the default passes only the text after ``</think>``).
    """
    return RubricVerifier(threshold)
