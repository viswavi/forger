"""Filter and grade stdin/stdout programs by running them on tests.

``by_tests`` keeps rows whose program passes its tests. ``program_verifier``
builds a ``metric`` for ``by_pass_at_k`` and ``by_comparative_difficulty``
that runs each sampled program on the row's tests.

Both run programs with ``pyforger.alter.programs`` (a subprocess per run, no
credentials in its environment, wall-clock and memory limits). That is NOT a
security sandbox: see ``pyforger.alter.programs``.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pyforger import _tracking
from pyforger.alter.programs import (
    Checker,
    last_code_block,
    outputs_match,
    run_programs,
    run_tests,
)
from pyforger.op import op
from pyforger.utils import get_field


@op(name="filter.by_tests")
def by_tests(
    items: list[dict],
    *,
    code_field: str,
    tests_field: str = "tests",
    min_pass_rate: float = 1.0,
    result_field: str | None = None,
    extract_code: Callable[[str], str | None] | None = None,
    checker: Checker = outputs_match,
    timeout_s: float = 10.0,
    memory_mb: int | None = 2048,
    command: list[str] | None = None,
    suffix: str = ".py",
    max_workers: int | None = None,
) -> list[dict]:
    """Keep rows whose program passes at least ``min_pass_rate`` of its tests.

    E.g. keep reference solutions that reproduce their statement's examples:
    ``by_tests(rows, code_field="solution", tests_field="examples")``. Every
    test of every row runs (``alter.run_programs``, cached on its own, so a new
    ``min_pass_rate`` runs nothing). The pass rate counts only tests with an
    ``"output"``; a row with no code or no such test is dropped. In a
    ``@step`` report the per-row score is the pass rate.

    Args:
        items: Input rows.
        code_field: Key holding the program source.
        tests_field: Key holding a list of ``{"input", "output", "args"?,
            "timeout_s"?}`` tests.
        min_pass_rate: Lowest pass rate kept (inclusive); 1.0 needs every test.
        result_field: When set, kept rows carry ``alter.run_programs``'s
            per-test results under this key.
        extract_code: Optional ``str -> str | None`` applied to the field
            first, e.g. ``last_code_block``.
        checker: ``(stdout, expected) -> bool``; default equal up to
            whitespace.
        timeout_s: Wall-clock limit per run, in seconds.
        memory_mb: Address-space limit per run (Linux only; None for none).
        command: Interpreter command with ``{file}`` for non-Python programs.
        suffix: File name suffix for the program.
        max_workers: Runs at a time (default ``min(8, cpu count)``).

    Returns:
        The kept rows, in input order.
    """
    if not 0.0 <= min_pass_rate <= 1.0:
        raise ValueError(f"min_pass_rate must be in [0, 1], got {min_pass_rate}")
    # Run only the code and tests, so rows that differ elsewhere reuse the
    # cached runs and the cache does not hold a second copy of every row.
    ran = run_programs(
        [
            {
                "code": get_field(item, code_field, "by_tests"),
                "tests": get_field(item, tests_field, "by_tests"),
            }
            for item in items
        ],
        code_field="code",
        tests_field="tests",
        result_field="_by_tests_results",
        extract_code=extract_code,
        checker=checker,
        timeout_s=timeout_s,
        memory_mb=memory_mb,
        command=command,
        suffix=suffix,
        max_workers=max_workers,
    )
    rates = [_pass_rate(row["_by_tests_results"]) for row in ran]
    _tracking.record_scores("pass rate", rates, keep="ge", threshold=min_pass_rate)
    return [
        {**item, result_field: row["_by_tests_results"]}
        if result_field is not None
        else item
        for item, row, rate in zip(items, ran, rates)
        if rate is not None and rate >= min_pass_rate
    ]


def _pass_rate(results: list[dict] | None) -> float | None:
    graded = [r["passed"] for r in results or [] if r["passed"] is not None]
    return sum(graded) / len(graded) if graded else None


def program_verifier(
    tests_field: str = "tests",
    *,
    extract_code: Callable[[str], str | None] = last_code_block,
    checker: Checker = outputs_match,
    timeout_s: float = 10.0,
    memory_mb: int | None = 2048,
    command: list[str] | None = None,
    suffix: str = ".py",
) -> Callable[[str, dict], bool | None]:
    """A ``metric`` for the difficulty filters that runs programs on tests.

    ``filter.by_pass_at_k(rows, ..., metric=filter.program_verifier("tests"))``
    takes the last fenced Python block from each sample's answer (after its
    ``<think>`` block), runs it on ``row["tests"]`` in order, and grades it
    True only if every test with an ``"output"`` passes. It stops at the first
    failure. An answer without a code block grades False; a row without tests
    grades None (not gradable). The difficulty filter runs several verifier
    calls at a time, so each call runs its tests one after another.

    Args:
        tests_field: Key holding the row's tests (see ``alter.run_programs``).
        extract_code: Takes the program from the prediction; default
            ``last_code_block``. Returning None grades False.
        checker: ``(stdout, expected) -> bool``; default equal up to
            whitespace.
        timeout_s: Wall-clock limit per test run, in seconds (a test's own
            ``"timeout_s"`` overrides it).
        memory_mb: Address-space limit per run (Linux only; None for none).
        command: Interpreter command with ``{file}`` for non-Python programs.
        suffix: File name suffix for the program.

    Returns:
        A ``(prediction, row) -> bool | None`` verifier with a ``cache_key()``,
        so equal settings share cached grades.
    """
    return _ProgramVerifier(
        tests_field,
        extract_code,
        checker,
        dict(timeout_s=timeout_s, memory_mb=memory_mb, command=command, suffix=suffix),
    )


class _ProgramVerifier:
    """The callable ``program_verifier`` returns; stateless, so thread-safe."""

    def __init__(
        self,
        tests_field: str,
        extract_code: Callable[[str], str | None],
        checker: Checker,
        run_kwargs: dict[str, Any],
    ) -> None:
        self.tests_field = tests_field
        self.extract_code = extract_code
        self.checker = checker
        self.run_kwargs = run_kwargs

    def cache_key(self) -> tuple:
        """Settings only, so equal verifiers share cached grades."""
        return (
            "program_verifier",
            self.tests_field,
            self.extract_code,
            self.checker,
            sorted(self.run_kwargs.items()),
        )

    def __repr__(self) -> str:
        """Show the tests field, e.g. ``program_verifier('tests')``."""
        return f"program_verifier({self.tests_field!r})"

    def __call__(self, prediction: str, row: dict) -> bool | None:
        """Grade one prediction: True when every graded test passes."""
        tests = [
            t
            for t in get_field(row, self.tests_field, "program_verifier") or []
            if t.get("output") is not None
        ]
        if not tests:
            return None
        code = self.extract_code(prediction)
        if not code or not code.strip():
            return False
        results = run_tests(
            code,
            tests,
            checker=self.checker,
            stop_on_failure=True,
            max_chars=0,
            **self.run_kwargs,
        )
        return len(results) == len(tests) and all(r["passed"] for r in results)
