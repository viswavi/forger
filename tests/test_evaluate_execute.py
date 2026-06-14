"""Tests for pyforger.evaluate.execute.

The executor runs untrusted code in a subprocess. Tests use trivially-correct
or trivially-incorrect Python to exercise the four outcome states
(``pass`` / ``fail`` / ``timeout`` / ``error``) without touching a model.
"""
from __future__ import annotations

import pytest

from pyforger.evaluate.execute import score_program


# A tiny HumanEval-shaped triple. The prompt declares the signature + docstring;
# the test checks the function adds correctly.
PROMPT = 'def add(x, y):\n    """Return x + y."""\n'
TEST = (
    "def check(candidate):\n"
    "    assert candidate(1, 2) == 3\n"
    "    assert candidate(-1, 1) == 0\n"
)
ENTRY = "add"


class TestOptIn:
    """The executor must refuse to run unless explicitly opted in."""

    def test_default_is_disabled(self):
        with pytest.raises(RuntimeError) as exc:
            score_program(PROMPT, "    return x + y", TEST, ENTRY)
        assert "allow_code_execution" in str(exc.value)


class TestOutcomes:
    """All four outcome states are reachable with simple programs."""

    def test_correct_solution_passes(self):
        out = score_program(
            PROMPT,
            "    return x + y",
            TEST,
            ENTRY,
            timeout_s=5,
            allow_code_execution=True,
        )
        assert out.outcome == "pass", f"stderr was: {out.stderr!r}"
        assert out.passed is True
        assert out.returncode == 0

    def test_wrong_solution_fails_via_assertion(self):
        out = score_program(
            PROMPT,
            "    return x - y",
            TEST,
            ENTRY,
            timeout_s=5,
            allow_code_execution=True,
        )
        assert out.outcome == "fail"
        assert out.passed is False
        assert "AssertionError" in out.stderr

    def test_syntax_error_in_body_is_error_not_fail(self):
        out = score_program(
            PROMPT,
            "    this is not valid python",
            TEST,
            ENTRY,
            timeout_s=5,
            allow_code_execution=True,
        )
        assert out.outcome == "error"
        assert "AssertionError" not in out.stderr

    def test_infinite_loop_times_out(self):
        out = score_program(
            PROMPT,
            "    while True:\n        pass\n",
            TEST,
            ENTRY,
            timeout_s=2,
            allow_code_execution=True,
        )
        assert out.outcome == "timeout"
        assert out.returncode is None
        assert out.duration_s >= 2.0


class TestEnvironmentIsolation:
    """The subprocess does not inherit secrets like HF_HOME."""

    def test_env_is_scrubbed(self, monkeypatch):
        # Write a probe that reports whether a sentinel env var leaked in.
        monkeypatch.setenv("HF_TOKEN", "should-not-leak")
        prompt = ""
        body = (
            "import os\n"
            "leaked = os.environ.get('HF_TOKEN')\n"
            "assert leaked is None, f'leaked={leaked!r}'\n"
        )
        # No `check` call needed; exit status alone tells us.
        out = score_program(
            prompt,
            body,
            "def check(candidate):\n    pass\n",
            "leaked",
            timeout_s=5,
            allow_code_execution=True,
        )
        # The probe asserts; the entry-point name doesn't matter because
        # `leaked` is not callable. We only care that the import-time
        # assertion passed (returncode 0 reaches the check call, which then
        # fails with TypeError). Our outcome will therefore be "error" —
        # that's fine; what we want is the absence of an AssertionError
        # mentioning HF_TOKEN.
        assert "leaked='should-not-leak'" not in out.stderr
