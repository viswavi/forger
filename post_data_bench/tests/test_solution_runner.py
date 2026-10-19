"""rl.solution_runner runs solutions as grade()'s run_program/call_method do."""
import importlib.util
import unittest

from coding.rl.solution_runner import (
    RLSandboxRunner,
    rubric_verifier,
    run_solution,
    score_targets,
)

DOUBLE = "print(int(input()) * 2)"
GRADE = (
    "def grade(prompt, response, ground_truth):\n"
    "    code = response.split('```python')[-1].split('```')[0]\n"
    "    ok, out = run_program(code, '2\\n')\n"
    "    return 1.0 if ok and out.strip() == '4' else 0.0"
)
THINKS = (
    "def grade(prompt, response, ground_truth):\n"
    "    return 1.0 if '</think>' in response else 0.0"
)


def target(code):
    """A ground truth whose answer is `code` in a python fence."""
    return f"<think>Reasoning</think>```python\n{code}\n```"


class Student:
    """A model whose samples of every prompt solve it, then fail it."""

    def cache_key(self):
        """Stable key for Forger's op cache."""
        return ("Student",)

    def generate_batch(self, prompts, n=1, **kwargs):
        """Two whole completions per prompt, thinking included."""
        self.prompts = prompts
        return [[target(DOUBLE), target("print(5)")][:n] for _ in prompts]


class RunSolutionTests(unittest.TestCase):
    """Scripts and class-based solutions in the RL solution sandbox."""

    def test_script_passes_and_fails(self):
        """Scripts pass when stdout equals the expected output up to whitespace."""
        results = run_solution(
            DOUBLE, [{"input": "2\n", "output": "4"}, {"input": "2\n", "output": "5"}]
        )
        self.assertEqual([r["passed"] for r in results], [True, False])
        self.assertEqual(results[0]["stdout"].strip(), "4")

    def test_call_compares_the_json_normalized_value(self):
        """A returned tuple equals the list it becomes in grade()."""
        code = "class Solution:\n    def pair(self, a):\n        return (a, a)"
        results = run_solution(
            code, [{"call": ["Solution", "pair", 3], "output": [3, 3]}]
        )
        self.assertTrue(results[0]["passed"])

    def test_refused_constructs_fail_with_the_reason(self):
        """__slots__ does not compile; str.format fails when called."""
        slots = (
            "class Solution:\n    __slots__ = ()\n    def f(self):\n        return 1"
        )
        result = run_solution(slots, [{"call": ["Solution", "f"], "output": 1}])[0]
        self.assertFalse(result["passed"])
        self.assertIn("__slots__", result["stderr"])
        result = run_solution("print('{}'.format(1))", [{"input": "", "output": "1"}])[
            0
        ]
        self.assertEqual(
            (result["passed"], result["stderr"]), (False, "NotImplementedError")
        )

    def test_main_guard_is_stripped_for_scripts(self):
        """As in grade()'s run_program."""
        code = "def main():\n    print(7)\nif __name__ == '__main__':\n    main()"
        self.assertTrue(run_solution(code, [{"input": "", "output": "7"}])[0]["passed"])

    def test_runner_results_feed_forger(self):
        """Without "output" nothing is graded: the runner leaves that to Forger."""
        results = RLSandboxRunner()(DOUBLE, [{"input": "3\n"}])
        self.assertEqual(results[0]["status"], "ok")
        self.assertNotIn("passed", results[0])


class ScoreTargetsTests(unittest.TestCase):
    """score_targets runs each row's code criteria on its own target."""

    def test_scores_code_criteria_and_skips_prompt_criteria(self):
        """A target using a refused construct scores 0 against its rubric."""
        rows = [
            {
                "prompt": "Double it",
                "ground_truth": target(DOUBLE),
                "rubric": [[{"code": GRADE}, 1.0], [{"prompt": "Clear?"}, 1.0]],
            },
            {
                "prompt": "Double it",
                "ground_truth": target("def _f(x):\n    return x * 2\nprint(_f(2))"),
                "rubric": [[{"code": GRADE}, 1.0]],
            },
        ]
        scores = score_targets(rows)
        self.assertEqual(
            scores[0], {"score": 1.0, "criteria": [1.0, None], "prompt_weight": 1.0}
        )
        self.assertEqual(scores[1]["score"], 0.0)


class RubricVerifierTests(unittest.TestCase):
    """rubric_verifier grades a sampled response with the row's own rubric."""

    ROW = {
        "prompt": [{"role": "user", "content": "Double it"}],
        "ground_truth": target(DOUBLE),
        "rubric": [[{"code": GRADE}, 3.0], [{"code": THINKS}, 1.0]],
    }

    def test_passes_at_the_threshold_on_the_whole_response(self):
        """The thinking counts, as in RL: without it the second criterion fails."""
        verify = rubric_verifier()
        self.assertTrue(verify(target(DOUBLE), self.ROW))
        self.assertFalse(verify(f"```python\n{DOUBLE}\n```", self.ROW))
        self.assertTrue(rubric_verifier(0.75)(f"```python\n{DOUBLE}\n```", self.ROW))
        self.assertFalse(verify(target("print(5)"), self.ROW))

    def test_matches_score_targets(self):
        """Same weights and normalization as the target check."""
        response = f"```python\n{DOUBLE}\n```"
        score = score_targets([{**self.ROW, "ground_truth": response}])[0]["score"]
        self.assertEqual(score, 0.75)
        self.assertFalse(rubric_verifier(0.76)(response, self.ROW))

    def test_none_without_code_criteria_or_a_valid_rubric(self):
        """Prompt criteria need the judge; a broken rubric cannot be graded."""
        verify = rubric_verifier()
        judged = {**self.ROW, "rubric": [[{"prompt": "Clear?"}, 1.0]]}
        self.assertIsNone(verify(target(DOUBLE), judged))
        broken = {**self.ROW, "rubric": [[{"code": "x = 1"}, 1.0]]}
        self.assertIsNone(verify(target(DOUBLE), broken))

    @unittest.skipUnless(importlib.util.find_spec("pyforger"), "needs pyforger")
    def test_whole_completion_reaches_it_through_by_pass_at_k(self):
        """The chat messages go out as they are; the thinking reaches the rubric."""
        from pyforger.filter import by_pass_at_k

        student = Student()
        rows = by_pass_at_k(
            [self.ROW],
            model=student,
            input_field="prompt",
            metric=rubric_verifier(),
            extract_answer=lambda c: c,
            k=2,
            min_rate=0.0,
            max_rate=1.0,
            outcomes_field="grades",
            _no_cache=True,
        )
        self.assertEqual(rows[0]["grades"], {"model": [True, False]})
        self.assertEqual(student.prompts, [self.ROW["prompt"]])


if __name__ == "__main__":
    unittest.main()
