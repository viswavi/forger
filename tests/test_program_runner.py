"""runner= on the program ops, and self_taught_reasoner's verifier=."""

from __future__ import annotations

import threading

import pytest
from fixtures import MockModel

from pyforger.alter import run_programs, self_taught_reasoner
from pyforger.alter.programs import last_code_block, run_tests
from pyforger.filter import by_tests, program_verifier


class EvalRunner:
    """Runs `code` as a Python expression of `x` per test; records concurrency.

    A test is {"input": "3", "output": "6"} (stdout = str(eval)), or
    {"call": 3, "output": 6} (graded here, on the value).
    """

    def __init__(self):
        """Start with no calls."""
        self.lock = threading.Lock()
        self.calls: list[str] = []

    def cache_key(self) -> tuple:
        """Stable key for @op caching."""
        return ("EvalRunner",)

    def __call__(self, code, tests):
        """One result per test."""
        with self.lock:
            self.calls.append(code)
        results = []
        for test in tests:
            try:
                if "call" in test:
                    value = eval(code, {"x": test["call"]})
                    results.append({"status": "ok", "passed": value == test["output"]})
                else:
                    value = eval(code, {"x": int(test["input"])})
                    results.append({"status": "ok", "stdout": str(value)})
            except Exception as exc:
                results.append(
                    {"status": "runtime_error", "stderr": type(exc).__name__}
                )
        return results


TESTS = [{"input": "3", "output": "6"}, {"input": "4", "output": "8"}]


class TestRunProgramsWithRunner:
    """run_programs grades a runner's results like its own."""

    def test_results_are_graded_and_completed(self):
        """The checker grades stdout; missing keys are filled in."""
        rows = [{"code": "x * 2", "tests": TESTS}, {"code": "x + 3", "tests": TESTS}]
        out = run_programs(rows, code_field="code", runner=EvalRunner())
        assert [[r["passed"] for r in row["test_results"]] for row in out] == [
            [True, True],
            [True, False],
        ]
        wrong = out[1]["test_results"][1]
        assert wrong["status"] == "wrong_answer" and wrong["stdout"] == "7"
        assert set(wrong) >= {"exit_code", "timed_out", "stderr", "seconds"}

    def test_runner_grades_calls_itself(self):
        """A runner's "passed" is kept; an error fails the test."""
        tests = [{"call": 3, "output": 6}, {"call": "a", "output": 1}]
        out = run_programs(
            [{"code": "x * 2", "tests": tests}], code_field="code", runner=EvalRunner()
        )
        assert [r["passed"] for r in out[0]["test_results"]] == [True, False]

    def test_missing_code_gets_none_and_is_not_run(self):
        """As without a runner: no code, no results."""
        runner = EvalRunner()
        out = run_programs(
            [{"code": "", "tests": TESTS}, {"code": "x", "tests": []}],
            code_field="code",
            runner=runner,
        )
        assert [row["test_results"] for row in out] == [None, []]
        assert runner.calls == []

    def test_wrong_result_count_raises(self):
        """A runner must return one result per test."""

        class Short(EvalRunner):
            def __call__(self, code, tests):
                return super().__call__(code, tests)[:1]

        with pytest.raises(ValueError, match="returned 1 results for 2 tests"):
            run_programs(
                [{"code": "x", "tests": TESTS}], code_field="code", runner=Short()
            )

    def test_run_tests_stops_after_first_failure(self):
        """stop_on_failure cuts the runner's results."""
        tests = [{"input": "1", "output": "9"}] + TESTS
        results = run_tests("x * 2", tests, runner=EvalRunner(), stop_on_failure=True)
        assert [r["passed"] for r in results] == [False]


class TestByTestsAndVerifierWithRunner:
    """by_tests and program_verifier take runner= too."""

    def test_by_tests(self):
        """Rows whose code passes every test are kept."""
        rows = [{"code": "x * 2", "tests": TESTS}, {"code": "x + 3", "tests": TESTS}]
        assert by_tests(rows, code_field="code", runner=EvalRunner()) == rows[:1]

    def test_program_verifier(self):
        """The verifier extracts the code block and runs it with the runner."""
        verify = program_verifier("tests", runner=EvalRunner())
        assert verify("```python\nx * 2\n```", {"tests": TESTS}) is True
        assert verify("```python\nx + 3\n```", {"tests": TESTS}) is False


class TestSelfTaughtReasonerVerifier:
    """verifier= sees each candidate row and runs in parallel."""

    def test_keeps_answers_the_verifier_accepts(self):
        """Only passing samples are kept, and reference_field is not needed."""
        model = MockModel(
            responses=[["```python\nx * 2\n```", "```python\nx + 3\n```"]]
        )
        kept = self_taught_reasoner(
            [{"q": "double", "tests": TESTS}],
            "{q}",
            model,
            n=2,
            verifier=program_verifier("tests", runner=EvalRunner()),
        )
        assert [last_code_block(row["answer"]) for row in kept] == ["x * 2\n"]
        assert "final_answer" not in kept[0]

    def test_verifier_gets_answer_and_row(self):
        """The answer is the text after the reasoning; the row has the item's fields."""
        seen = []

        def verifier(answer, row):
            seen.append((answer, row["q"]))
            return None  # not gradable: dropped

        model = MockModel(responses=[["<think>hmm</think>done"]])
        assert (
            self_taught_reasoner([{"q": "x"}], "{q}", model, n=1, verifier=verifier)
            == []
        )
        assert seen == [("done", "x")]

    def test_filter_fn_and_verifier_are_exclusive(self):
        """Passing both is an error."""
        with pytest.raises(ValueError, match="filter_fn or verifier"):
            self_taught_reasoner(
                [{"q": "x"}],
                "{q}",
                MockModel(responses=[["a"]]),
                n=1,
                filter_fn=lambda rows: rows,
                verifier=lambda answer, row: True,
            )
