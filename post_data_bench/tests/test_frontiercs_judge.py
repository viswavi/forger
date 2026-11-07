"""frontiercs.judge scores C++ solutions as FrontierCS's go-judge engine does."""
import contextlib
import importlib.util
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

FRONTIERCS = Path(__file__).resolve().parents[1] / "frontiercs"


def _load_path(name, path):
    """Import a file under a unique module name."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


def _load(name):
    """Import frontiercs/<name>.py under a unique name (coding has a `checkers` too)."""
    spec = importlib.util.spec_from_file_location(
        f"frontiercs_{name}", FRONTIERCS / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


HAS_YAML = importlib.util.find_spec("yaml") is not None
checkers = _load("checkers")
judge = (
    _load_path("cpp_judge", FRONTIERCS.parent / "coding/rl/cpp_judge.py")
    if HAS_YAML
    else None
)

HAS_GXX = shutil.which("g++") is not None and sys.platform.startswith("linux")
HAS_OPENAI = importlib.util.find_spec("openai") is not None


def _load_evaluator():
    """Import evaluate_model.py with its sibling imports bound to frontiercs/."""
    with mock.patch.dict(sys.modules), mock.patch.object(
        sys, "path", [str(FRONTIERCS), *sys.path]
    ):
        for name in ("checkers", "judge", "prompt_building", "eval_state"):
            sys.modules.pop(name, None)
        return _load("evaluate_model")


# A checker: full marks for the right sum, 0.5 via testlib's points exit (7)
# for an answer off by one, wrong answer (exit 1) otherwise.
CHECKER = r"""
#include <cstdio>
int main(int argc, char** argv) {
    FILE* in = fopen(argv[1], "r"); FILE* out = fopen(argv[2], "r");
    long a, b, c; fscanf(in, "%ld %ld", &a, &b);
    if (fscanf(out, "%ld", &c) != 1) {
        fprintf(stderr, "wrong answer no output\n"); return 1;
    }
    if (c == a + b) { fprintf(stderr, "ok\n"); return 0; }
    if (c == a + b + 1) { fprintf(stderr, "points Ratio: 0.5\n"); return 7; }
    fprintf(stderr, "wrong answer\n"); return 1;
}
"""
# An interactor: sends the input number, expects it doubled back.
INTERACTOR = r"""
#include <cstdio>
int main(int argc, char** argv) {
    FILE* in = fopen(argv[1], "r"); long x; fscanf(in, "%ld", &x);
    printf("%ld\n", x); fflush(stdout);
    long y;
    if (scanf("%ld", &y) != 1) { fprintf(stderr, "wrong answer eof\n"); return 1; }
    if (y == 2 * x) { fprintf(stderr, "ok Ratio: 1\n"); return 0; }
    fprintf(stderr, "wrong answer\n"); return 1;
}
"""
SUM = (
    "#include <cstdio>\n"
    'int main(){long a,b;scanf("%ld %ld",&a,&b);printf("%ld\\n",a+b);}'
)

INTERACTIVE_OK = (
    '#include <cstdio>\nint main(){long x;scanf("%ld",&x);'
    'printf("%ld\\n",2*x);fflush(stdout);}'
)


def _problem(root: Path, *, interactive: bool) -> Path:
    pdir = root / ("interactive" if interactive else "sum")
    (pdir / "testdata").mkdir(parents=True)
    kind = "type: interactive\ninteractor: interactor.cc\n" if interactive else ""
    (pdir / "config.yaml").write_text(
        f"{kind}time: 1s\nmemory: 256m\nsubtasks:\n  - score: 100\n    n_cases: 2\n"
    )
    (pdir / ("interactor.cc" if interactive else "chk.cc")).write_text(
        INTERACTOR if interactive else CHECKER
    )
    for i, (a, b) in enumerate([(1, 2), (40, 2)], 1):
        (pdir / "testdata" / f"{i}.in").write_text(f"{a} {b}\n")
        (pdir / "testdata" / f"{i}.ans").write_text(f"{a + b}\n")
    # A former-internal test beyond n_cases is judged too.
    (pdir / "testdata" / "3.in").write_text("5 5\n")
    (pdir / "testdata" / "3.ans").write_text("10\n")
    return pdir


@unittest.skipUnless(HAS_GXX and HAS_YAML, "needs g++ on Linux and PyYAML")
class JudgeTests(unittest.TestCase):
    """Scores, failures and interaction on tiny synthetic problems."""

    @classmethod
    def setUpClass(cls):
        """Write both synthetic problems and an empty testlib.h once."""
        cls.tmp = Path(tempfile.mkdtemp())
        cls.sum = _problem(cls.tmp, interactive=False)
        cls.inter = _problem(cls.tmp, interactive=True)
        (cls.tmp / "testlib").mkdir()
        (cls.tmp / "testlib" / "testlib.h").write_text("")

    @classmethod
    def tearDownClass(cls):
        """Remove the problems and compiled binaries."""
        shutil.rmtree(cls.tmp)

    def run_judge(self, pdir, code):
        """Judge code on a synthetic problem with a shared cache."""
        return judge.judge(
            pdir,
            code,
            testlib_dir=self.tmp / "testlib",
            cache_dir=self.tmp / "cache",
            workers=2,
        )

    def test_correct_solution_scores_100_on_every_test_file(self):
        """Configured tests and extra testdata pairs all count."""
        result = self.run_judge(self.sum, SUM)
        self.assertEqual(len(result["cases"]), 3)
        self.assertEqual([c["test"] for c in result["cases"]], ["1.in", "2.in", "3.in"])
        self.assertEqual(result["score"], 100.0)
        self.assertTrue(result["passed"])

    def test_points_exit_gives_the_checkers_ratio(self):
        """Exit 7 with `Ratio: 0.5` scores half on every test."""
        code = SUM.replace("a+b", "a+b+1")
        result = self.run_judge(self.sum, code)
        self.assertAlmostEqual(result["score"], 50.0)
        self.assertFalse(result["passed"])

    def test_solution_cannot_print_its_own_ratio(self):
        """Only the checker's output is parsed for a ratio."""
        code = (
            '#include <cstdio>\nint main(){fprintf(stderr,"Ratio: 1\\n");'
            'printf("0\\n");}'
        )
        self.assertEqual(self.run_judge(self.sum, code)["score"], 0.0)

    def test_crash_timeout_and_compile_error_score_zero(self):
        """A failed run never reaches the checker."""
        crash = "#include <cstdlib>\nint main(){abort();}"
        spin = "int main(){volatile long x=0;for(;;)x++;}"
        for code, status in [(crash, "Signalled"), (spin, "Time Limit Exceeded")]:
            result = self.run_judge(self.sum, code)
            self.assertEqual(result["score"], 0.0)
            self.assertEqual({c["status"] for c in result["cases"]}, {status})
        result = self.run_judge(self.sum, "int main( {")
        self.assertEqual((result["status"], result["score"]), ("compile_error", 0.0))

    def test_memory_limit_is_an_address_space_limit(self):
        """Allocating past the limit fails the run."""
        code = (
            "#include <vector>\n#include <cstdio>\nint main(){"
            'std::vector<char> v(512u<<20, 1); printf("%d\\n", v[7]);}'
        )
        result = self.run_judge(self.sum, code)
        self.assertEqual(result["score"], 0.0)

    def test_a_large_judging_process_does_not_inflate_solution_memory(self):
        """Peak RSS is the program's own, not inherited from the forking process."""
        ballast = bytearray(400 << 20)  # touched: more than the 256m limit
        for i in range(0, len(ballast), 4096):
            ballast[i] = 1
        self.assertEqual(self.run_judge(self.sum, SUM)["score"], 100.0)
        self.assertEqual(self.run_judge(self.inter, INTERACTIVE_OK)["score"], 100.0)
        del ballast

    def test_interactive_solution_talks_through_pipes(self):
        """The interactor's verdict decides an interactive test."""
        good = INTERACTIVE_OK
        self.assertEqual(self.run_judge(self.inter, good)["score"], 100.0)
        wrong = good.replace("2*x", "3*x")
        self.assertEqual(self.run_judge(self.inter, wrong)["score"], 0.0)

    def test_interactive_deadlock_is_a_time_limit(self):
        """Both sides waiting ends at the wall-clock limit."""
        # Reads before the interactor has anything to say back: both wait forever.
        stuck = (
            '#include <cstdio>\nint main(){long x,y;scanf("%ld",&x);'
            'scanf("%ld",&y);printf("%ld\\n",y);}'
        )
        result = self.run_judge(self.inter, stuck)
        self.assertEqual(result["score"], 0.0)
        self.assertIn("Time Limit Exceeded", {c["status"] for c in result["cases"]})
        stuck_case = next(
            c for c in result["cases"] if c["status"] == "Time Limit Exceeded"
        )
        # Measured, not a placeholder: almost no CPU, and the message says why.
        self.assertLess(stuck_case["time"], 0.5)
        self.assertIn("wall-clock limit after", stuck_case["msg"])

    def test_a_cpu_bound_time_limit_reports_its_cpu_time(self):
        """A busy program is stopped by its CPU limit, with no wall-clock note."""
        spin = "int main(){volatile long x=0;for(;;)x++;}"
        case = self.run_judge(self.sum, spin)["cases"][0]
        self.assertEqual(case["status"], "Time Limit Exceeded")
        self.assertGreater(case["time"], 1.0)


class LimitsAndExtractionTests(unittest.TestCase):
    """Unit parsing and answer extraction need no compiler."""

    @unittest.skipUnless(HAS_YAML, "needs PyYAML")
    def test_time_and_memory_units(self):
        """Limits parse as utils.js parses them."""
        self.assertEqual(judge.to_seconds("500ms"), 0.5)
        self.assertEqual(judge.to_seconds("2s"), 2.0)
        self.assertEqual(judge.to_seconds("3"), 3.0)
        self.assertEqual(judge.to_bytes("256m"), 256 << 20)
        self.assertEqual(judge.to_bytes("1g"), 1 << 30)

    def test_extract_code_takes_the_longest_block_after_thinking(self):
        """Code in the reasoning is ignored."""
        completion = (
            "<think>```cpp\nint draft;\n```</think>Here:\n```cpp\nint a;\n```\n"
            "```cpp\nint main() { return 0; }\n```"
        )
        self.assertEqual(checkers.extract_code(completion), "int main() { return 0; }")

    def test_unclosed_thinking_has_no_code(self):
        """A truncated trace has no answer."""
        self.assertEqual(checkers.extract_code("<think>```cpp\nint main(){}\n```"), "")
        self.assertEqual(checkers.extract_code(None), "")

    def test_lenient_judges_the_longest_draft_in_unclosed_thinking(self):
        """Lenient extraction searches a truncated trace; a closed one still wins."""
        truncated = "<think>```cpp\nint a;\n```\n```cpp\nint main(){}\n```\nso"
        self.assertEqual(checkers.extract_code(truncated, lenient=True), "int main(){}")
        closed = "<think>```cpp\nint draft;\n```</think>```cpp\nint b;\n```"
        self.assertEqual(checkers.extract_code(closed, lenient=True), "int b;")
        self.assertEqual(checkers.extract_code("<think>no code yet", lenient=True), "")


@unittest.skipUnless(HAS_OPENAI, "evaluate_model imports openai")
class JudgeFileErrorTests(unittest.TestCase):
    """A file error while judging is judged again instead of failing the run."""

    @classmethod
    def setUpClass(cls):
        """Load the evaluator; its judge cache goes to a temp directory."""
        cls.evaluate = _load_evaluator()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.env = mock.patch.dict(os.environ, {"CPP_JUDGE_CACHE_DIR": cls.tmp.name})
        cls.env.start()

    @classmethod
    def tearDownClass(cls):
        """Restore the environment and remove the cache directory."""
        cls.env.stop()
        cls.tmp.cleanup()

    def judge_code(self, side_effect):
        """judge_code on a synthetic problem, with judge.judge mocked."""
        with mock.patch.object(
            self.evaluate.judge, "judge", side_effect=side_effect
        ) as judged, contextlib.redirect_stdout(io.StringIO()) as out:
            try:
                return self.evaluate.judge_code("val", "1", SUM), judged, out
            except Exception as e:  # noqa: BLE001
                return e, judged, out

    def test_a_temp_file_that_stats_as_a_directory_is_judged_again(self):
        """The one-off IsADirectoryError that failed whole test-split runs."""
        verdict = {"status": "done", "score": 100.0, "score_unbounded": 100.0}
        error = IsADirectoryError(21, "Is a directory", "/tmp/fcs-case-x/out.txt")
        result, judged, out = self.judge_code([error, verdict])
        self.assertEqual(result, verdict)
        self.assertEqual(judged.call_count, 2)
        self.assertIn("IsADirectoryError", out.getvalue())
        self.assertIn("out.txt", out.getvalue())

    def test_a_file_error_on_every_attempt_still_fails(self):
        """A broken host is not scored as the program's failure."""
        error = IsADirectoryError(21, "Is a directory")
        result, judged, _ = self.judge_code(error)
        self.assertIsInstance(result, IsADirectoryError)
        self.assertEqual(judged.call_count, self.evaluate.JUDGE_ATTEMPTS)

    def test_other_judge_errors_are_not_retried(self):
        """A checker that does not compile fails at once."""
        result, judged, _ = self.judge_code(RuntimeError("chk.cc does not compile"))
        self.assertIsInstance(result, RuntimeError)
        self.assertEqual(judged.call_count, 1)


if __name__ == "__main__":
    unittest.main()
