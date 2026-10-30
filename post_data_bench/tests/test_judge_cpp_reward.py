"""grade()'s judge_cpp: C++ judged outside the rubric worker, as RL rewards run it."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from coding.rl import rubric_rm
from coding.rl.rubric_sandbox import Sandbox
from coding.rl.solution_runner import score_targets

RL = Path(__file__).resolve().parents[1] / "coding" / "rl"
HAS_GXX = (
    sys.platform.startswith("linux")
    and shutil.which("g++") is not None
    and shutil.which("gcc") is not None
)

SUM = (
    "#include <cstdio>\n"
    'int main(){long a,b;scanf("%ld %ld",&a,&b);printf("%ld\\n",a+b);}'
)
# Full marks for the sum, half through testlib's points exit for the sum plus one.
POINTS_CHECKER = r"""
#include "testlib.h"
int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);
    long long want = ans.readLong(), got = ouf.readLong();
    if (got == want) quitf(_ok, "correct");
    if (got == want + 1) quitp(0.5, "Ratio: 0.5");
    quitf(_wa, "expected %lld, got %lld", want, got);
}
"""
# Sends the input number and expects it doubled back.
INTERACTOR = r"""
#include "testlib.h"
#include <iostream>
int main(int argc, char* argv[]) {
    registerInteraction(argc, argv);
    long long x = inf.readLong();
    std::cout << x << std::endl;
    long long y = ouf.readLong();
    if (y == 2 * x) quitf(_ok, "doubled");
    quitf(_wa, "expected %lld, got %lld", 2 * x, y);
}
"""
DOUBLE = (
    '#include <cstdio>\nint main(){long x;scanf("%ld",&x);'
    'printf("%ld\\n",2*x);fflush(stdout);}'
)
SPIN = "int main(){volatile long x=0;for(;;)x++;}"
TESTS = [{"input": "1 2\n", "answer": "3\n"}, {"input": "40 2\n", "answer": "42\n"}]


def criterion(tests, **kwargs):
    """grade(): judge the response's last cpp block; score it, or check the result.

    With ``expect``, return 1 when judge_cpp's (score, statuses) has that score
    and its first status starts with the given prefix, else 0.
    """
    expect = kwargs.pop("expect", None)
    call = f"judge_cpp(code, {tests!r}"
    call += "".join(f", {k}={v!r}" for k, v in kwargs.items()) + ")"
    body = (
        "import re\n"
        "def grade(prompt, response, ground_truth):\n"
        "    blocks = re.findall(r'```cpp\\n(.*?)```', response, re.S)\n"
        "    code = blocks[-1] if blocks else ''\n"
        f"    score, statuses = {call}\n"
    )
    if expect is None:
        return body + "    return score\n"
    value, prefix = expect
    return (
        body + f"    ok = score == {value!r} and statuses[0].startswith({prefix!r})\n"
        "    return 1.0 if ok else 0.0\n"
    )


def response(code):
    """A completion whose answer is ``code`` in a cpp fence."""
    return f"<think>Plan</think>Here:\n```cpp\n{code}\n```"


def running_solutions():
    """Judged programs still alive (none once a call returns), by their argv."""
    alive = []
    for proc in Path("/proc").iterdir():
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if any(arg.endswith(b"judge_cpp/solution/main") for arg in argv):
            alive.append(proc.name)
    return alive


@unittest.skipUnless(HAS_GXX, "needs g++ and gcc on Linux")
class JudgeCppRewardTests(unittest.TestCase):
    """Criteria that call judge_cpp, scored by rubric_rm.code_score."""

    @classmethod
    def setUpClass(cls):
        """A private compile cache, warmed so later criteria time only themselves."""
        cls.cache = tempfile.mkdtemp(prefix="judge-cpp-test-")
        cls.env = mock.patch.dict(os.environ, {"CPP_JUDGE_CACHE_DIR": cls.cache})
        cls.env.start()
        Sandbox().judge_cpp(SUM, TESTS, checker=POINTS_CHECKER, timeout_s=60)
        Sandbox().judge_cpp(
            DOUBLE, [{"input": "3", "answer": ""}], interactor=INTERACTOR, timeout_s=60
        )

    @classmethod
    def tearDownClass(cls):
        """Drop the cache."""
        cls.env.stop()
        shutil.rmtree(cls.cache, ignore_errors=True)

    def score(self, code, crit):
        """One RL reward of the response ``code`` under criterion ``crit``."""
        return rubric_rm.code_score(crit, [], response(code), "")

    def test_correct_program_scores_one_and_whitespace_does_not_matter(self):
        """Without a checker, output equals the answer word for word."""
        spaced = SUM.replace('"%ld\\n"', '"  %ld \\n\\n"')
        self.assertEqual(self.score(SUM, criterion(TESTS)), 1.0)
        self.assertEqual(self.score(spaced, criterion(TESTS)), 1.0)

    def test_wrong_program_scores_zero(self):
        """A wrong output, or one with an extra word, fails the test."""
        self.assertEqual(self.score(SUM.replace("a+b", "a-b"), criterion(TESTS)), 0.0)
        extra = SUM.replace('"%ld\\n"', '"%ld 0\\n"')
        self.assertEqual(self.score(extra, criterion(TESTS)), 0.0)
        one_wrong = [TESTS[0], {"input": "1 1\n", "answer": "3\n"}]
        self.assertEqual(self.score(SUM, criterion(one_wrong)), 0.5)

    def test_points_checker_gives_partial_scores(self):
        """A testlib checker's `Ratio: 0.5` scores half a test."""
        crit = criterion(TESTS, checker=POINTS_CHECKER)
        self.assertEqual(self.score(SUM, crit), 1.0)
        off_by_one = SUM.replace("a+b", "a+b+1")
        self.assertEqual(self.score(off_by_one, crit), 0.5)
        self.assertEqual(
            Sandbox().judge_cpp(off_by_one, TESTS, checker=POINTS_CHECKER),
            (0.5, ["Partially Correct", "Partially Correct"]),
        )

    def test_interactive_program_talks_to_the_interactor(self):
        """A testlib interactor decides each test of an interactive program."""
        tests = [{"input": "3", "answer": ""}, {"input": "21", "answer": ""}]
        crit = criterion(tests, interactor=INTERACTOR)
        self.assertEqual(self.score(DOUBLE, crit), 1.0)
        self.assertEqual(self.score(DOUBLE.replace("2*x", "3*x"), crit), 0.0)

    def test_compile_error_scores_zero_with_the_compiler_message(self):
        """The response's compile error is a status, not a crash."""
        crit = criterion(TESTS, expect=(0.0, "Compile Error: "))
        self.assertEqual(self.score("int main( {", crit), 1.0)
        score, statuses = Sandbox().judge_cpp("int main( {", TESTS)
        self.assertEqual(score, 0.0)
        self.assertIn("error", statuses[0])
        bad_checker = Sandbox().judge_cpp(SUM, TESTS, checker="int main( {")
        self.assertTrue(bad_checker[1][0].startswith("Checker Compile Error: "))

    def test_invalid_arguments_score_zero_without_raising(self):
        """grade() keeps running after each bad call and sees why it failed."""
        bad = [
            ("tests", "1 2"),
            ("tests", [{"input": "1 2", "output": "3"}]),
            ("tests", [{"input": 1, "answer": "3"}]),
            ("tests", [TESTS[0]] * 51),
            ("time_limit", 0),
            ("time_limit", 11),
            ("time_limit", True),
            ("memory_mb", 8),
            ("memory_mb", 4096),
        ]
        for name, value in bad:
            args = {"tests": TESTS, name: value}
            tests = args.pop("tests")
            crit = criterion(tests, expect=(0.0, "Invalid Argument: "), **args)
            self.assertEqual(self.score(SUM, crit), 1.0, (name, value))
        both = criterion(
            TESTS,
            checker=POINTS_CHECKER,
            interactor=INTERACTOR,
            expect=(0.0, "Invalid Argument: "),
        )
        self.assertEqual(self.score(SUM, both), 1.0)
        not_json = (
            "def grade(prompt, response, ground_truth):\n"
            "    score, statuses = judge_cpp('int main(){}', [{'input': {1}}])\n"
            "    return 1.0 if statuses[0].startswith('Invalid Argument') else 0.0\n"
        )
        self.assertEqual(rubric_rm.code_score(not_json, [], "", ""), 1.0)
        self.assertEqual(Sandbox().judge_cpp(None, TESTS)[0], 0.0)

    def test_budget_exhaustion_scores_zero_and_kills_the_judging(self):
        """Judging that outlasts the criterion's 20 s ends with it: no orphans."""
        tests = [{"input": "", "answer": ""}] * 12
        start = time.monotonic()
        value = self.score(SPIN, criterion(tests, time_limit=10))
        elapsed = time.monotonic() - start
        self.assertEqual(value, 0.0)
        self.assertLess(elapsed, rubric_rm.WORKER_TIMEOUT_S)
        self.assertEqual(running_solutions(), [])
        result = Sandbox().judge_cpp(SPIN, tests, time_limit=10, timeout_s=3)
        self.assertEqual(result[0], 0.0)
        self.assertTrue(result[1][0].startswith("Budget Exceeded"))
        self.assertEqual(running_solutions(), [])

    def test_time_limit_is_a_test_status(self):
        """A short time limit fails a slow program inside the budget."""
        crit = criterion(TESTS, time_limit=0.2, expect=(0.0, "Time Limit Exceeded"))
        self.assertEqual(self.score(SPIN, crit), 1.0)

    def test_score_targets_and_direct_worker_run_judge_cpp(self):
        """The agent-facing checks score judge_cpp criteria as RL does."""
        rows = [
            {
                "prompt": "Add two numbers.",
                "ground_truth": response(SUM),
                "rubric": [[{"code": criterion(TESTS, checker=POINTS_CHECKER)}, 1.0]],
            }
        ]
        self.assertEqual(score_targets(rows)[0]["score"], 1.0)
        payload = json.dumps(
            {
                "code": criterion(TESTS),
                "prompt": [],
                "response": response(SUM),
                "ground_truth": "",
            }
        )
        out = subprocess.run(
            [sys.executable, str(RL / "rubric_worker.py")],
            input=payload,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(json.loads(out.stdout), {"score": 1.0}, out.stderr)

    def test_broken_judge_stops_the_reward(self):
        """An unusable judge (no g++, no cache) raises instead of scoring 0."""
        with mock.patch.dict(os.environ, {"CPP_JUDGE_CACHE_DIR": "/proc/forbidden"}):
            with self.assertRaises(RuntimeError):
                self.score(SUM, criterion(TESTS))


class JudgeCppWithoutChannelTests(unittest.TestCase):
    """Argument checks need no compiler."""

    def test_invalid_arguments_in_process(self):
        """Sandbox().judge_cpp rejects what grade()'s judge_cpp rejects."""
        cases = [
            ((SUM, "tests"), "Invalid Argument: tests"),
            ((SUM, TESTS, None, None, float("nan")), "Invalid Argument: arguments"),
            ((SUM, TESTS, "chk", "int"), "Invalid Argument: give a checker"),
        ]
        for args, prefix in cases:
            score, statuses = Sandbox().judge_cpp(*args)
            self.assertEqual(score, 0.0)
            self.assertTrue(statuses[0].startswith(prefix), statuses)


if __name__ == "__main__":
    unittest.main()
