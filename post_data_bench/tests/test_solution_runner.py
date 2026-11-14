"""rl.solution_runner runs solutions as grade()'s run_program/call_method do."""
import asyncio
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from coding.rl import rubric_rm
from coding.rl.solution_runner import (
    RLSandboxRunner,
    _messages,
    rubric_score,
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

    def test_negative_code_criterion_subtracts(self):
        """A met negative-weight code criterion lowers the score.

        prompt_weight counts only positive weight.
        """
        always = "def grade(p, r, g):\n    return 1.0"
        row = {
            "prompt": "Double it",
            "ground_truth": target(DOUBLE),
            "rubric": [
                [{"code": GRADE}, 4.0],
                [{"code": always}, -1.0],
                [{"prompt": "Mistake?"}, -3.0],
            ],
        }
        self.assertEqual(
            score_targets([row])[0],
            {"score": 0.75, "criteria": [1.0, 1.0, None], "prompt_weight": 0},
        )


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


class RubricScoreTests(unittest.TestCase):
    """rubric_score returns a response's weighted code score, for rubric_scores."""

    ROW = RubricVerifierTests.ROW

    def test_weighted_code_score_of_the_whole_response(self):
        """Same weights as score_targets: the missing thinking costs 1 of 4."""
        grade = rubric_score()
        self.assertEqual(grade(target(DOUBLE), self.ROW), 1.0)
        self.assertEqual(grade(f"```python\n{DOUBLE}\n```", self.ROW), 0.75)
        self.assertEqual(grade(target("print(5)"), self.ROW), 0.25)
        self.assertEqual(grade("", self.ROW), 0.0)

    def test_none_without_code_criteria_and_raises_on_a_broken_rubric(self):
        """Prompt criteria need the judge; a broken rubric is an error to report."""
        grade = rubric_score()
        judged = {**self.ROW, "rubric": [[{"prompt": "Clear?"}, 1.0]]}
        self.assertIsNone(grade(target(DOUBLE), judged))
        with self.assertRaises(Exception):
            grade(target(DOUBLE), {**self.ROW, "rubric": [[{"code": "x = 1"}, 1.0]]})

    def test_cache_key_is_stable_and_names_the_grader(self):
        """Two instances share cache entries; the verifier's are its own."""
        self.assertEqual(rubric_score().cache_key(), rubric_score().cache_key())
        self.assertNotEqual(rubric_score().cache_key(), rubric_verifier().cache_key())
        self.assertEqual(repr(rubric_score()), "rubric_score()")

    @unittest.skipUnless(importlib.util.find_spec("pyforger"), "needs pyforger")
    def test_grades_targets_samples_and_controls_in_rubric_scores(self):
        """Empty and random score 0 on a sound rubric; samples land in their bins."""
        from pyforger.inspect import rubric_scores

        rows = [
            {**self.ROW, "prompt": [{"role": "user", "content": f"Double it ({i})"}]}
            for i in range(3)
        ]
        report = rubric_scores(
            rows,
            rubric_score(),
            response_fields="ground_truth",
            model=Student(),
            k=2,
            _no_cache=True,
        )
        self.assertEqual(report.column("ground_truth").counts, [0, 0, 0, 0, 0, 3])
        self.assertEqual(report.column("sampled").counts, [0, 3, 0, 0, 0, 3])
        for name in ("empty", "random"):
            self.assertEqual(report.column(name).counts[0], 3)
        # Every row is the same task, so another row's target solves it too.
        self.assertEqual([f.split()[0] for f in report.flags], ["other_row"])


def judge_reply(score):
    """A LiteLLM judge reply giving ``score``, with token usage for the ledger."""
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=json.dumps({"score": score})),
            )
        ],
    )


class JudgedRubricScoreTests(unittest.TestCase):
    """rubric_score(judge=True) is the RL reward, prompt criteria included."""

    ROW = {
        **RubricVerifierTests.ROW,
        "rubric": [[{"code": GRADE}, 1.0], [{"prompt": "Explains it?"}, 3.0]],
    }

    def setUp(self):
        """A private judge ledger and credentials; no real judge calls."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.ledger = Path(temporary.name) / "judge_usage.json"
        environment = patch.dict(
            os.environ,
            {
                "JUDGE_TOKEN_USAGE_FILE": str(self.ledger),
                "JUDGE_TOKEN_UNIT_LIMIT": "75000000",
                "JUDGE_MODEL": rubric_rm.JUDGE_MODEL,
                "JUDGE_REASONING_EFFORT": "low",
                "PDB_TASK": "coding",
                "LITELLM_JUDGE_API_KEY": "test",
                "LITELLM_BASE_URL": "https://example.test",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)

    def judge(self, **kwargs):
        """Patch the judge call; kwargs as for AsyncMock."""
        return patch("litellm.acompletion", new=AsyncMock(**kwargs))

    def test_weights_code_and_judged_criteria_as_rl_does(self):
        """Code 1 x weight 1 and judge 0.5 x weight 3 give 0.625; default skips it."""
        with self.judge(return_value=judge_reply(0.5)) as completion:
            self.assertEqual(rubric_score(judge=True)(target(DOUBLE), self.ROW), 0.625)
            self.assertEqual(rubric_score()(target(DOUBLE), self.ROW), 1.0)
        self.assertEqual(completion.await_count, 1)
        sent = json.loads(completion.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(sent["prompt"], self.ROW["prompt"])
        self.assertEqual(sent["response"], target(DOUBLE))
        self.assertEqual(json.loads(self.ledger.read_text())["calls"], 1)

    def test_matches_custom_rm(self):
        """The same rubric, prompt, response and label give custom_rm's reward."""
        row = {**self.ROW, "prompt": "Double it"}  # a string prompt becomes a message
        sample = SimpleNamespace(
            response=target("print(5)"),
            label=row["ground_truth"],
            metadata={"rubric": row["rubric"], "original_prompt": _messages(row)},
        )
        with self.judge(return_value=judge_reply(1)):
            expected = asyncio.run(rubric_rm.custom_rm(None, sample))
            self.assertEqual(expected, 0.75)
            self.assertEqual(
                rubric_score(judge=True)(target("print(5)"), row), expected
            )

    def test_content_filter_blocks_leave_the_mean(self):
        """A blocked prompt criterion is unscored: the code criterion decides."""
        blocked = RuntimeError('Error code: 400 {"finish_reason":"content_filter"}')
        only_judged = {**self.ROW, "rubric": [[{"prompt": "Explains it?"}, 1.0]]}
        with self.judge(side_effect=blocked), patch("sys.stderr"):
            self.assertEqual(rubric_score(judge=True)(target(DOUBLE), self.ROW), 1.0)
            self.assertEqual(rubric_score(judge=True)(target(DOUBLE), only_judged), 0.0)
            self.assertIsNone(rubric_score()(target(DOUBLE), only_judged))

    def test_judge_failures_raise_instead_of_scoring_zero(self):
        """An unavailable judge is an error to report, never a 0 reward."""
        with self.judge(side_effect=RuntimeError("down")), patch.dict(
            os.environ, {"JUDGE_MAX_CONSECUTIVE_FAILURES": "1"}
        ), patch.object(rubric_rm.asyncio, "sleep", new=AsyncMock()), patch(
            "sys.stderr"
        ):
            with self.assertRaises(rubric_rm.JudgeUnavailable):
                rubric_score(judge=True)(target(DOUBLE), self.ROW)

    def test_works_inside_a_running_event_loop(self):
        """E.g. from a notebook cell: the reward runs on a loop of its own."""

        async def caller():
            return rubric_score(judge=True)(target(DOUBLE), self.ROW)

        with self.judge(return_value=judge_reply(1)):
            self.assertEqual(asyncio.run(caller()), 1.0)

    def test_cache_key_and_repr_name_the_judge(self):
        """Judged scores never share cache entries with code-only ones."""
        judged = rubric_score(judge=True)
        self.assertEqual(judged.cache_key(), rubric_score(judge=True).cache_key())
        self.assertNotEqual(judged.cache_key(), rubric_score().cache_key())
        self.assertIn(rubric_rm.JUDGE_MODEL, judged.cache_key())
        self.assertEqual(repr(judged), "rubric_score(judge=True)")
        glm = "litellm_proxy/azure_ai/FW-GLM-5.3"
        with patch.dict(os.environ, {"JUDGE_FALLBACK_MODEL": glm}):
            self.assertIn(glm, judged.cache_key())
        with patch.dict(os.environ):
            os.environ.pop("JUDGE_FALLBACK_MODEL", None)
            unset = judged.cache_key()
            os.environ["JUDGE_FALLBACK_MODEL"] = ""
            self.assertEqual(judged.cache_key(), unset)  # "" keeps the old keys


if __name__ == "__main__":
    unittest.main()


class SwappedArgumentTests(unittest.TestCase):
    """The graders take (response, row); swapped arguments raise, not score 0."""

    def test_swapped_arguments_raise(self):
        """A row passed as the response is a TypeError naming the order."""
        row = {"prompt": "p", "ground_truth": "g", "rubric": []}
        for grade in (rubric_score(), rubric_verifier()):
            with self.assertRaisesRegex(TypeError, r"grade\(response, row\)"):
                grade(row, "text")
