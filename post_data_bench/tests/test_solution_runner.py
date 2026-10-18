"""rl.solution_runner runs solutions as grade()'s run_program/call_method do."""
import unittest

from coding.rl.solution_runner import RLSandboxRunner, run_solution, score_targets

DOUBLE = "print(int(input()) * 2)"
GRADE = (
    "def grade(prompt, response, ground_truth):\n"
    "    code = response.split('```python')[-1].split('```')[0]\n"
    "    ok, out = run_program(code, '2\\n')\n"
    "    return 1.0 if ok and out.strip() == '4' else 0.0"
)


def target(code):
    """A ground truth whose answer is `code` in a python fence."""
    return f"<think>Reasoning</think>```python\n{code}\n```"


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


if __name__ == "__main__":
    unittest.main()
