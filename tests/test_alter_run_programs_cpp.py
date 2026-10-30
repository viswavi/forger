"""C++ programs and partial scores in run_programs, by_tests, and program_verifier."""
from __future__ import annotations

import os
import shutil

import pytest

from pyforger.alter import run_programs
from pyforger.alter.programs import (
    compile_program,
    last_code_block,
    run_tests,
    testlib_checker,
)
from pyforger.filter import by_tests, program_verifier

pytestmark = pytest.mark.skipif(
    os.name != "posix" or shutil.which("g++") is None, reason="needs POSIX and g++"
)

ADD = (
    "#include <iostream>\n"
    "int main() { long long a, b; std::cin >> a >> b; std::cout << a + b << '\\n'; }\n"
)
# Prints a + b - DROP, so DROP > 0 gives a partial answer and DROP < 0 too much.
OFF_BY = (
    "#include <iostream>\n"
    "int main() {{ long long a, b; std::cin >> a >> b; std::cout << a + b - ({}); }}\n"
)
BROKEN = "int main( {"
TESTS = [{"input": "1 2\n", "output": "3"}, {"input": "2 2\n", "output": "5"}]
# A checker with testlib's exit codes but no testlib.h: answers below the
# expected value get its fraction (_points, 7), above it nothing (_wa, 1).
CHECKER = r"""
#include <cstdio>
int main(int argc, char** argv) {
    long long want = 0, got = 0;
    FILE* ans = std::fopen(argv[3], "r");
    FILE* out = std::fopen(argv[2], "r");
    if (std::fscanf(ans, "%lld", &want) != 1) return 3;
    if (std::fscanf(out, "%lld", &got) != 1) return 2;
    if (got == want) return 0;
    if (got > want) return 1;
    std::printf("Ratio: %.4f\n", double(got) / double(want));
    return 7;
}
"""


class TestCompile:
    """compile_program builds a binary or returns the compiler's message."""

    def test_binary_or_error(self, tmp_path):
        """A good program gives a binary; a broken one names main.cpp."""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        binary, error = compile_program(ADD, str(tmp_path / "a"))
        assert error == "" and os.path.exists(binary)
        binary, error = compile_program(BROKEN, str(tmp_path / "b"))
        assert binary is None and error.startswith("main.cpp:")

    def test_unknown_language(self, tmp_path):
        """Only Python and C++ are known."""
        with pytest.raises(ValueError, match="language"):
            compile_program(ADD, str(tmp_path), "rust")


class TestRunPrograms:
    """language="cpp" compiles once per program and runs the binary per test."""

    def test_results_and_compile_errors(self):
        """Pass, wrong answer, and a compile error on every test of its row."""
        rows = run_programs(
            [{"code": ADD, "tests": TESTS}, {"code": BROKEN, "tests": TESTS}],
            code_field="code",
            language="cpp",
            _no_cache=True,
        )
        first, second = (row["test_results"] for row in rows)
        assert [(r["status"], r["passed"], r["score"]) for r in first] == [
            ("ok", True, 1.0),
            ("wrong_answer", False, 0.0),
        ]
        assert [r["status"] for r in second] == ["compile_error"] * 2
        assert second[0]["stderr"].startswith("main.cpp:")

    def test_ungraded_test_keeps_output(self):
        """A test without "output" records stdout, e.g. a brute force's answers."""
        rows = run_programs(
            [{"code": ADD, "tests": [{"input": "20 22"}]}],
            code_field="code",
            language="cpp",
            _no_cache=True,
        )
        result = rows[0]["test_results"][0]
        assert (result["passed"], result["score"], result["stdout"].strip()) == (
            None,
            None,
            "42",
        )

    def test_command_and_suffix_are_for_interpreted_programs(self):
        """A compiled program runs as its binary."""
        with pytest.raises(ValueError, match="compile_command"):
            run_programs(
                [{"code": ADD, "tests": TESTS}],
                code_field="code",
                language="cpp",
                command=["node", "{file}"],
                _no_cache=True,
            )


class TestScores:
    """Checkers may return partial scores; a test passes at 1."""

    def test_float_checker_and_test_argument(self):
        """A three-argument checker gets the test; floats become "partial"."""
        seen = []

        def half(stdout, expected, test):
            seen.append(test["input"])
            return 1.0 if stdout.split() == expected.split() else 0.5

        results = run_tests(ADD, TESTS, checker=half, language="cpp")
        assert [(r["status"], r["score"], r["passed"]) for r in results] == [
            ("ok", 1.0, True),
            ("partial", 0.5, False),
        ]
        assert seen == ["1 2\n", "2 2\n"]

    def test_testlib_checker(self):
        """Ratio on _points, 1 on _ok, 0 on _wa."""
        checker = testlib_checker(CHECKER)
        rows = run_programs(
            [
                {"code": OFF_BY.format(d), "tests": [{"input": "5 5", "output": "10"}]}
                for d in (0, 5, -1)
            ],
            code_field="code",
            language="cpp",
            checker=checker,
            _no_cache=True,
        )
        scores = [row["test_results"][0]["score"] for row in rows]
        assert scores == [1.0, 0.5, 0.0]

    def test_broken_checker_raises(self):
        """A checker that does not compile is an error, not a zero score."""
        with pytest.raises(RuntimeError, match="does not compile"):
            run_tests(ADD, TESTS[:1], checker=testlib_checker(BROKEN), language="cpp")

    def test_by_tests_uses_the_mean_score(self):
        """A row scoring 1 and 0.5 has pass rate 0.75."""
        checker = testlib_checker(CHECKER)
        tests = [{"input": "5 5", "output": "10"}, {"input": "4 4", "output": "8"}]
        rows = [{"code": OFF_BY.format(d), "tests": tests} for d in (0, 4)]
        kept = by_tests(
            rows,
            code_field="code",
            language="cpp",
            checker=checker,
            min_pass_rate=0.7,
            _no_cache=True,
        )
        # Off by 4: 6/10 and 4/8, mean 0.55, so only the exact program stays.
        assert [row["code"] for row in kept] == [OFF_BY.format(0)]


class TestProgramVerifier:
    """program_verifier grades C++ answers, all-pass or by mean score."""

    def test_cpp_block_and_all_pass(self):
        """The last C++ block is run; every graded test must pass."""
        verify = program_verifier(language="cpp")
        answer = f"Idea.\n```cpp\n{ADD}```\n"
        assert verify(answer, {"tests": TESTS[:1]}) is True
        assert verify(answer, {"tests": TESTS}) is False
        assert verify("no code", {"tests": TESTS}) is False

    def test_min_score(self):
        """Below 1, the mean score over every test decides."""
        checker = testlib_checker(CHECKER)
        row = {
            "tests": [{"input": "5 5", "output": "10"}, {"input": "4 4", "output": "8"}]
        }
        answer = "```cpp\n" + OFF_BY.format(2) + "```"  # 8/10 and 6/8: mean 0.775
        assert program_verifier(language="cpp", checker=checker, min_score=0.75)(
            answer, row
        )
        assert not program_verifier(language="cpp", checker=checker, min_score=0.8)(
            answer, row
        )

    def test_python_verifier_key_is_unchanged(self):
        """Python verifiers keep their cache keys (no language entries)."""
        key = program_verifier().cache_key()
        assert "language" not in str(key) and "min_score" not in str(key)


def test_last_code_block_by_language():
    """C++ tags select C++ blocks; untagged blocks count for both."""
    text = "```cpp\nint main(){}\n```\n```python\nprint(1)\n```"
    assert last_code_block(text, "cpp") == "int main(){}\n"
    assert last_code_block(text) == "print(1)\n"
    assert last_code_block("```\nx\n```", "cpp") == "x\n"
