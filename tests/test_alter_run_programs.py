"""Tests for running stdin/stdout programs: alter.run_programs, filter.by_tests."""
from __future__ import annotations

import os
import sys
import threading

import pytest

from pyforger import _tracking
from pyforger.alter import run_programs
from pyforger.alter.programs import (
    last_code_block,
    outputs_match,
    run_program,
    run_tests,
)
from pyforger.filter import by_tests, program_verifier
from pyforger.op import trace
from pyforger.steps import step

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX only")

ADD = "a, b = map(int, input().split())\nprint(a + b)\n"
TESTS = [{"input": "1 2\n", "output": "3"}, {"input": "2 2\n", "output": "5"}]


class TestRunProgram:
    """Running one script in a subprocess reports what happened."""

    def test_ok(self):
        """Stdin goes in, stdout comes back, exit code 0 is ok."""
        result = run_program(ADD, "4 5\n")
        assert result["status"] == "ok"
        assert result["stdout"] == "9\n"
        assert result["exit_code"] == 0
        assert result["timed_out"] is False

    def test_runtime_error_keeps_stderr(self):
        """A non-zero exit is a runtime error with the traceback in stderr."""
        result = run_program("raise ValueError('boom')")
        assert result["status"] == "runtime_error"
        assert result["exit_code"] == 1
        assert "ValueError: boom" in result["stderr"]

    def test_environment_has_no_credentials(self, monkeypatch):
        """The child sees PATH and the locale only, not the parent's variables."""
        monkeypatch.setenv("LITELLM_API_KEY", "secret")
        result = run_program("import os\nprint(sorted(os.environ))")
        assert result["stdout"].strip() == "['LANG', 'LC_ALL', 'PATH']"

    def test_runs_in_a_fresh_directory(self):
        """Each run starts in its own empty temporary directory, removed after."""
        result = run_program("import os\nprint(os.getcwd())\nprint(os.listdir('.'))")
        cwd, listing = result["stdout"].splitlines()
        assert cwd != os.getcwd()
        assert "main.py" in listing
        assert not os.path.exists(cwd)

    def test_args(self):
        """The args become the program's command-line arguments, e.g. a seed."""
        result = run_program("import sys\nprint(sys.argv[1:])", args=["7", "small"])
        assert result["stdout"].strip() == "['7', 'small']"

    def test_timeout_kills_the_process_group(self, tmp_path):
        """A timed-out run is killed along with any process it started."""
        marker = tmp_path / "survived"
        grandchild = f"import time; time.sleep(1.5); open({str(marker)!r}, 'w')"
        code = (
            "import subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, '-c', {grandchild!r}])\n"
            "time.sleep(30)\n"
        )
        result = run_program(code, timeout_s=0.5)
        assert result["status"] == "timeout"
        assert result["exit_code"] is None
        assert result["timed_out"] is True
        threading.Event().wait(2.0)
        assert not marker.exists()

    def test_output_limit(self):
        """A program that floods stdout stops with output_limit."""
        code = "import sys\nwhile True:\n    sys.stdout.write('x' * 4096)\n"
        result = run_program(code, max_output_mb=1, timeout_s=20)
        assert result["status"] == "output_limit"
        assert len(result["stdout"]) == 1 << 20

    @pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS is Linux only")
    def test_memory_limit(self):
        """Allocating past memory_mb fails the run."""
        result = run_program("x = bytearray(1 << 30)\nprint(len(x))", memory_mb=256)
        assert result["status"] == "runtime_error"
        assert "MemoryError" in result["stderr"]

    def test_other_interpreter(self):
        """A command runs non-Python programs, with {file} for the program."""
        result = run_program(
            'read a\necho "got $a"\n', "hi\n", command=["sh", "{file}"], suffix=".sh"
        )
        assert result["stdout"] == "got hi\n"

    def test_safe_from_threads(self):
        """No signals: runs work from worker threads at the same time."""
        results: list[dict] = []

        def go(i):
            results.append(run_program(ADD, f"{i} {i}\n"))

        threads = [threading.Thread(target=go, args=(i,)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert sorted(int(r["stdout"]) for r in results) == [0, 2, 4, 6, 8, 10]


class TestRunTests:
    """Tests run in order and are graded."""

    def test_grades_and_wrong_answer(self):
        """Exit 0 with the wrong output is a wrong answer."""
        results = run_tests(ADD, TESTS)
        assert [(r["status"], r["passed"]) for r in results] == [
            ("ok", True),
            ("wrong_answer", False),
        ]

    def test_stop_on_failure(self):
        """stop_on_failure skips the tests after the first failure."""
        results = run_tests(ADD, list(reversed(TESTS)), stop_on_failure=True)
        assert len(results) == 1

    def test_test_without_output_is_not_graded(self):
        """A test without "output" only records the run, e.g. a brute force's answer."""
        (result,) = run_tests(ADD, [{"input": "1 1\n"}])
        assert result["status"] == "ok" and result["passed"] is None
        assert result["stdout"] == "2\n"

    def test_per_test_timeout(self):
        """A test's timeout_s overrides the default."""
        code = "import time\ntime.sleep(float(input()))\nprint('done')"
        tests = [{"input": "1", "output": "done", "timeout_s": 0.3}]
        (result,) = run_tests(code, tests, timeout_s=30)
        assert result["status"] == "timeout"

    def test_custom_checker(self):
        """A checker replaces whitespace-insensitive equality."""
        tests = [{"input": "1 2", "output": "3.0001"}]
        close = lambda got, want: abs(float(got) - float(want)) < 1e-3  # noqa: E731
        assert run_tests(ADD, tests)[0]["passed"] is False
        assert run_tests(ADD, tests, checker=close)[0]["passed"] is True

    def test_max_chars(self):
        """Kept stdout is cut to max_chars (stderr keeps its end); None keeps all."""
        code = "print('x' * 50)\nraise SystemExit('the end')"
        (cut,) = run_tests(code, [{"input": ""}], max_chars=5)
        assert cut["stdout"] == "x" * 5
        assert cut["stderr"] == " end\n"
        (whole,) = run_tests(code, [{"input": ""}], max_chars=None)
        assert whole["stdout"] == "x" * 50 + "\n"


@pytest.mark.parametrize(
    "actual,expected,same",
    [("1 2\n3\n", "1 2 3", True), ("1  2", "1 2\n", True), ("12", "1 2", False)],
)
def test_outputs_match(actual, expected, same):
    """The default checker ignores whitespace differences only."""
    assert outputs_match(actual, expected) is same


@pytest.mark.parametrize(
    "text,code",
    [
        ("Idea.\n```python\nprint(1)\n```", "print(1)\n"),
        ("```\nA\n```\n```py\nB\n```", "B\n"),
        ("```cpp\nint main(){}\n```", None),
        ("no code", None),
    ],
)
def test_last_code_block(text, code):
    """The last untagged or Python-tagged fenced block, else None."""
    assert last_code_block(text) == code


class TestRunProgramsOp:
    """The alter op records per-test results on every row."""

    ROWS = [
        {"id": 0, "code": ADD, "tests": TESTS},
        {"id": 1, "code": None, "tests": TESTS},
        {"id": 2, "code": "```python\nprint(3)\n```", "tests": TESTS[:1]},
    ]

    def test_results_per_row(self):
        """Each row gets one result per test; a row without code gets None."""
        out = run_programs(self.ROWS, code_field="code")
        assert [r["passed"] for r in out[0]["test_results"]] == [True, False]
        assert out[1]["test_results"] is None
        assert out[2]["test_results"][0]["status"] == "runtime_error"  # the fence
        assert "test_results" not in self.ROWS[0]

    def test_extract_code(self):
        """extract_code takes the program out of a model's answer first."""
        out = run_programs(
            self.ROWS[2:], code_field="code", extract_code=last_code_block
        )
        assert out[0]["test_results"][0]["passed"] is True

    def test_cached(self):
        """A second identical call is a cache hit."""
        run_programs(self.ROWS, code_field="code")
        run_programs(self.ROWS, code_field="code")
        hits = [r.cache_hit for r in trace() if r.op_name == "alter.run_programs"]
        assert hits == [False, True]

    def test_max_workers_validated(self):
        """max_workers must be positive."""
        with pytest.raises(ValueError):
            run_programs(self.ROWS, code_field="code", max_workers=-1)


class TestByTests:
    """filter.by_tests keeps rows whose program passes enough tests."""

    ROWS = [
        {"id": "all", "code": ADD, "tests": TESTS[:1]},
        {"id": "half", "code": ADD, "tests": TESTS},
        {"id": "none", "code": "print(0)", "tests": TESTS},
        {"id": "no_code", "code": "", "tests": TESTS},
        {"id": "ungraded", "code": ADD, "tests": [{"input": "1 1"}]},
    ]

    def test_min_pass_rate(self):
        """Default keeps only rows passing every test; lower bounds keep more."""
        assert [r["id"] for r in by_tests(self.ROWS, code_field="code")] == ["all"]
        kept = by_tests(self.ROWS, code_field="code", min_pass_rate=0.5)
        assert [r["id"] for r in kept] == ["all", "half"]
        kept = by_tests(self.ROWS, code_field="code", min_pass_rate=0.0)
        assert [r["id"] for r in kept] == ["all", "half", "none"]

    def test_new_bound_runs_nothing(self):
        """The runs are cached apart from min_pass_rate."""
        by_tests(self.ROWS, code_field="code")
        by_tests(self.ROWS, code_field="code", min_pass_rate=0.5)
        hits = [r.cache_hit for r in trace() if r.op_name == "alter.run_programs"]
        assert hits == [False, True]

    def test_result_field(self):
        """result_field keeps the per-test results on kept rows."""
        kept = by_tests(self.ROWS, code_field="code", result_field="runs")
        assert kept[0]["runs"][0]["passed"] is True
        assert "runs" not in self.ROWS[0]

    def test_scores_in_step(self, monkeypatch):
        """The step report gets each row's pass rate, None when ungradable."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")

        @step
        def passing(rows):
            return by_tests(rows, code_field="code", min_pass_rate=0.5)

        passing(self.ROWS)
        scores = _tracking.step_records()["passing"].ops[0].scores
        assert scores is not None
        assert scores.values == [1.0, 0.5, 0.0, None, None]
        assert (scores.keep, scores.threshold) == ("ge", 0.5)

    def test_bad_rate(self):
        """min_pass_rate must be in [0, 1]."""
        with pytest.raises(ValueError):
            by_tests(self.ROWS, code_field="code", min_pass_rate=1.5)


class TestProgramVerifier:
    """The verifier grades one prediction for the difficulty filters."""

    ROW = {"tests": TESTS[:1] + [{"input": "5 5", "output": "10"}]}

    def test_grades(self):
        """True when all tests pass, False on a wrong program or no code."""
        verify = program_verifier()
        assert verify(f"Sum them.\n```python\n{ADD}```", self.ROW) is True
        assert verify("```python\nprint(3)\n```", self.ROW) is False
        assert verify("I cannot solve this.", self.ROW) is False

    def test_no_tests_is_ungradable(self):
        """A row without graded tests grades None (excluded, not failed)."""
        verify = program_verifier()
        assert verify(f"```python\n{ADD}```", {"tests": []}) is None
        assert verify(f"```python\n{ADD}```", {"tests": [{"input": "1 1"}]}) is None

    def test_cache_key_reflects_settings(self):
        """Equal settings share a cache key; different ones do not."""
        assert program_verifier().cache_key() == program_verifier().cache_key()
        assert program_verifier().cache_key() != program_verifier("t").cache_key()
        assert repr(program_verifier("t")) == "program_verifier('t')"
