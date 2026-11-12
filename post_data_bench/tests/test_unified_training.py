"""Regression tests for unified training."""
import argparse
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from coding.rl import agent_training as training
from coding.rl import rubric_data as data
from coding.rl import rubric_rm as rm
from coding.rl import rubric_worker as worker

ROOT = Path(__file__).resolve().parents[1]


def rows():
    """Create distinct shared-triple fixtures for submission tests."""
    return [
        {
            "prompt": f"Question {i}",
            "ground_truth": "<think>Reasoning</think>Answer",
            "rubric": [
                [
                    {
                        ("code"): (
                            "def grade(prompt, response, ground_truth):\n    return 1.0"
                        )
                    },
                    1,
                ]
            ],
        }
        for i in range(20)
    ]


def write_rows(path, values):
    """Write fixture rows as newline-delimited JSON."""
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in values))


def read(path):
    """Load newline-delimited JSON fixture records."""
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


class DatasetTests(unittest.TestCase):
    """Regression checks for dataset."""

    def test_split_and_sft_rl_identity(self):
        """Verify split and sft rl identity."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_rows(root / "input", rows())
            data.compile_data(root / "input", root / "a", sft=True)
            train, val, sft = [
                read(root / "a" / name)
                for name in ("train.jsonl", "val.jsonl", "sft.jsonl")
            ]
            self.assertEqual((len(train), len(val), len(sft)), (18, 2, 18))
            self.assertEqual(
                [r["prompt"] for r in train], [r["messages"][:-1] for r in sft]
            )
            self.assertFalse(
                {json.dumps(r["prompt"]) for r in train}
                & {json.dumps(r["prompt"]) for r in val}
            )
            self.assertNotIn("rubric", json.dumps([r["messages"] for r in sft]))
            write_rows(root / "input", list(reversed(rows())))
            data.compile_data(root / "input", root / "b")
            self.assertEqual(
                (root / "a/train.jsonl").read_bytes(),
                (root / "b/train.jsonl").read_bytes(),
            )
            self.assertFalse((root / "b/sft.jsonl").exists())

    def test_rl_rows_need_no_ground_truth(self):
        """RL rows are prompt + rubric; ground_truth is optional free text."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bare = [{key: row[key] for key in ("prompt", "rubric")} for row in rows()]
            bare[0]["ground_truth"] = "a reference answer, no thinking trace"
            write_rows(root / "input", bare)
            data.compile_data(root / "input", root / "a")
            labels = {
                r["label"]
                for name in ("train.jsonl", "val.jsonl")
                for r in read(root / "a" / name)
            }
            self.assertEqual(labels, {"", "a reference answer, no thinking trace"})
            with self.assertRaisesRegex(ValueError, "SFT needs"):
                data.compile_data(root / "input", root / "b", sft=True)
            with self.assertRaisesRegex(ValueError, "may have"):
                data.validate_row({**bare[1], "response": "x"})

    def test_reject_duplicate_and_small(self):
        """Verify reject duplicate and small."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input"
            for values in (rows()[:1], rows() + [rows()[0]]):
                write_rows(path, values)
                with self.assertRaises(ValueError):
                    data.compile_data(path, Path(tmp) / "out")

    def test_invalid_weights_and_union(self):
        """Verify invalid weights and union."""
        for rubric in (
            [],
            [[{"code": "x", "prompt": "x"}, 1]],
            [[{"prompt": "x"}, -1]],
            [[{"prompt": "x"}, float("nan")]],
            [[{"prompt": "x"}, 0]],
        ):
            with self.subTest(rubric=rubric), self.assertRaises(ValueError):
                data.validate_rubric(rubric)

    def test_answer_only_on_prompt_criteria(self):
        """answer_only is a bool on prompt criteria only, in either key order."""
        for criterion in (
            {"prompt": "x", "answer_only": True},
            {"answer_only": False, "prompt": "x"},
        ):
            with self.subTest(criterion=criterion):
                data.validate_rubric([[criterion, 1]])
        for criterion, message in (
            (
                {"code": "def grade(p, r, g):\n    return 1.0", "answer_only": True},
                "only to",
            ),
            ({"prompt": "x", "answer_only": "yes"}, "true or false"),
            ({"prompt": "x", "answer_only": 1}, "true or false"),
            ({"prompt": " ", "answer_only": True}, "nonempty"),
            ({"answer_only": True}, "exactly one"),
        ):
            with self.subTest(criterion=criterion), self.assertRaisesRegex(
                ValueError, message
            ):
                data.validate_rubric([[criterion, 1]])

    def test_thinking_and_overlength(self):
        """Verify thinking and overlength."""
        with self.assertRaises(ValueError):
            data.validate_sft_target("answer")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input"
            write_rows(path, rows())
            calls = []

            def template(messages, **kwargs):
                calls.append((messages, kwargs))
                return [0] * 101

            tokenizer = SimpleNamespace(apply_chat_template=template)
            for sft in (False, True):
                with self.subTest(sft=sft), self.assertRaisesRegex(
                    ValueError, "exceeds"
                ):
                    data.compile_data(
                        path, Path(tmp) / "out", tokenizer, max_length=100, sft=sft
                    )
                self.assertFalse((Path(tmp) / "out").exists())
            # RL checks the prompt alone; SFT checks prompt plus target.
            (rl_messages, rl_kwargs), (sft_messages, sft_kwargs) = calls
            self.assertEqual(rl_messages[-1]["role"], "user")
            self.assertTrue(rl_kwargs["add_generation_prompt"])
            self.assertEqual(sft_messages[-1]["role"], "assistant")
            self.assertFalse(sft_kwargs["add_generation_prompt"])


class CodeTests(unittest.TestCase):
    """Regression checks for code."""

    def test_named_grade_and_helpers(self):
        """Verify named grade and helpers."""
        source = (
            "def grade(prompt, response, ground_truth):\n    return "
            "helper(response == ground_truth)\ndef helper(value):\n    "
            "return float(value)"
        )
        self.assertEqual(rm.code_score(source, [], "a", "a"), 1)
        self.assertEqual(rm.code_score(source, [], "a", "b"), 0)

    def test_grade_can_call_final_answer(self):
        """final_answer in grade() follows rubric_rm.final_answer's rule."""
        source = (
            "def grade(prompt, response, ground_truth):\n"
            "    answer = final_answer(response)\n"
            "    return 1.0 if answer == final_answer(ground_truth) else 0.0\n"
        )
        gold = "<think>work</think> 42 "
        for response, expected in (
            ("<think>a</think>b</think> 42", 1.0),
            ("<think>a</think> 41", 0.0),
            ("<think>cut off before 42", 0.0),
        ):
            with self.subTest(response=response):
                self.assertEqual(rm.code_score(source, [], response, gold), expected)
        for text in ("<think>a</think>b</think> c ", "<think>open", " plain ", None):
            with self.subTest(text=text):
                self.assertEqual(worker.final_answer(text), rm.final_answer(text))

    def test_forbidden_access_and_invalid_score(self):
        """Verify forbidden access and invalid score."""
        for source in (
            "import os\ndef grade(p,r,g):\n    return float(bool(os.environ))",
            "def grade(p,r,g):\n    return open('/etc/passwd').read()",
            "def grade(p,r,g):\n    return 12",
            "def grade(p,r,g):\n    return r.__class__",
        ):
            with self.subTest(source=source):
                self.assertEqual(rm.code_score(source, [], "a", "b"), 0)

    def test_timeout(self):
        """Verify timeout."""
        self.assertEqual(
            rm.code_score(
                "def grade(p,r,g):\n    while True:\n        pass", [], "a", "a"
            ),
            0,
        )


class SolutionExecutionTests(unittest.TestCase):
    """Code criteria can run the response's code against tests, safely."""

    STDIN_GRADER = (
        "import re\n"
        "TESTS = [('1 2\\n', '3'), ('5 7\\n', '12')]\n"
        "def grade(prompt, response, ground_truth):\n"
        "    answer = response.split('</think>')[-1]\n"
        "    blocks = re.findall(r'```python\\n(.*?)```', answer, re.S)\n"
        "    if not blocks:\n"
        "        return 0.0\n"
        "    passed = 0\n"
        "    for stdin, expected in TESTS:\n"
        "        ok, out = run_program(blocks[-1], stdin)\n"
        "        passed += ok and out.strip() == expected\n"
        "    return passed / len(TESTS)\n"
    )

    def response(self, code):
        """A thinking-mode response whose answer is code in a python fence."""
        return "<think>add</think>```python\n" + code + "\n```"

    def test_runs_stdin_programs_and_scores_tests(self):
        """Correct, partially correct, and failing programs get graded by tests."""
        for code, expected in (
            ("a, b = map(int, input().split())\nprint(a + b)", 1.0),
            ("a, b = map(int, input().split())\nprint(3)", 0.5),
            ("print(int(input()) + )", 0.0),
            ("while True:\n    pass", 0.0),
            ("import os\nprint(os.getcwd())", 0.0),
        ):
            with self.subTest(code=code):
                self.assertEqual(
                    rm.code_score(self.STDIN_GRADER, [], self.response(code), ""),
                    expected,
                )

    def test_common_solution_idioms_pass(self):
        """Star-args, sys.exit, bytearray, and class helpers are valid solutions."""
        for code in (
            "a, b = map(int, input().split())\nprint(*[a + b])",
            "import sys\na, b = map(int, input().split())\nprint(a + b)\nsys.exit(0)",
            "import sys\na, b = map(int, input().split())\nprint(a + b)\nsys.exit()",
            "buf = bytearray(input(), 'ascii')\n"
            "print(sum(map(int, bytes(buf).split())))",
            (
                "class S:\n    @staticmethod\n"
                "    def add(a, b):\n        return a + b\n"
                "print(S.add(*map(int, input().split())))"
            ),
            "a, b = map(int, input().split())\n"
            "print(a + b if hasattr(a, 'real') else 0)",
        ):
            with self.subTest(code=code):
                self.assertEqual(
                    rm.code_score(self.STDIN_GRADER, [], self.response(code), ""),
                    1.0,
                )

    def test_grade_can_pass_star_args_to_call_method(self):
        """call_method(code, cls, method, *args) works from inside grade()."""
        grader = (
            "CODE = ('class Solution:\\n    def add(self, a, b):\\n'\n"
            "        '        return a + b\\n')\n"
            "def grade(prompt, response, ground_truth):\n"
            "    args = [2, 3]\n"
            "    ok, result = call_method(CODE, 'Solution', 'add', *args)\n"
            "    return float(ok and result == 5)\n"
        )
        self.assertEqual(rm.code_score(grader, [], "", ""), 1.0)

    def test_guarded_getattr_still_refuses_private_names(self):
        """getattr/hasattr go through the attribute guard."""
        grader = (
            "def grade(prompt, response, ground_truth):\n"
            "    return float(not hasattr(response, '__class__'))\n"
        )
        self.assertEqual(rm.code_score(grader, [], "", ""), 1.0)
        leak = (
            "def grade(prompt, response, ground_truth):\n"
            "    return float(getattr(response, '__class__') is str)\n"
        )
        self.assertEqual(rm.code_score(leak, [], "", ""), 0.0)

    def test_worker_gets_single_threaded_blas(self):
        """The worker env pins BLAS to one thread (numpy under the 512MB limit)."""
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            self.assertEqual(rm._WORKER_ENV[name], "1")

    def test_response_crash_scores_zero_but_broken_worker_raises(self):
        """A worker crash the trivial probe does not reproduce is a 0, not a crash."""
        crashed = SimpleNamespace(returncode=1, stdout="", stderr="OpenBLAS error")
        healthy = SimpleNamespace(returncode=0, stdout='{"score": 1.0}', stderr="")
        with patch.object(rm, "_HEALTHY_WORKER", []), patch.object(
            rm, "_run_worker", side_effect=[crashed, healthy]
        ):
            self.assertEqual(rm.code_score(self.STDIN_GRADER, [], "x", ""), 0.0)
        with patch.object(rm, "_HEALTHY_WORKER", []), patch.object(
            rm, "_run_worker", side_effect=[crashed, crashed]
        ):
            with self.assertRaisesRegex(RuntimeError, "rubric worker failed"):
                rm.code_score(self.STDIN_GRADER, [], "x", "")

    def test_timeout_is_per_run(self):
        """One looping test fails alone; later runs still get their own budget."""
        grader = (
            "def grade(prompt, response, ground_truth):\n"
            "    slow = run_program('while True:\\n    pass')\n"
            "    fast = run_program('print(2 + 2)')\n"
            "    return float(\n"
            "        slow == (False, 'SandboxTimeout') and fast == (True, '4\\n')\n"
            "    )\n"
        )
        self.assertEqual(rm.code_score(grader, [], "", ""), 1.0)

    def test_response_cannot_forge_scores(self):
        """Printing a score, or returning an always-equal object, earns nothing."""
        forged_output = "import sys\nprint('{\"score\": 1.0}')\nsys.exit(0)"
        self.assertEqual(
            rm.code_score(self.STDIN_GRADER, [], self.response(forged_output), ""), 0.0
        )
        cheat = (
            "class Anything:\n"
            "    def __eq__(self, other):\n"
            "        return True\n"
            "class Solution:\n"
            "    def add(self, a, b):\n"
            "        return Anything()\n"
        )
        honest = (
            "class Solution:\n"
            "    def add(self, a, b):\n"
            "        return [a + b, (a, b)]\n"
        )
        grader = (
            "CODE = %r\n"
            "def grade(prompt, response, ground_truth):\n"
            "    ok, result = call_method(CODE, 'Solution', 'add', 2, 3)\n"
            "    return float(ok and result == [5, [2, 3]])\n"
        )
        self.assertEqual(rm.code_score(grader % cheat, [], "", ""), 0.0)
        self.assertEqual(rm.code_score(grader % honest, [], "", ""), 1.0)


class RewardTests(unittest.IsolatedAsyncioTestCase):
    """Regression checks for reward."""

    def setUp(self):
        """Isolate persistent judge usage and environment settings."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.usage_file = Path(temporary.name) / "judge_usage.json"
        environment = patch.dict(
            os.environ,
            {
                "JUDGE_TOKEN_USAGE_FILE": str(self.usage_file),
                "JUDGE_TOKEN_UNIT_LIMIT": "75000000",
                "JUDGE_MODEL": rm.JUDGE_MODEL,
                "JUDGE_REASONING_EFFORT": "low",
                "PDB_TASK": "coding",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)

    async def test_mixed_weighted_reward_and_original_prompt(self):
        """Verify mixed weighted reward and original prompt."""
        sample = SimpleNamespace(
            response="<think>trace</think>answer",
            label="gold",
            prompt="templated",
            metadata={
                "original_prompt": [{"role": "user", "content": "original"}],
                "rubric": [
                    [{"code": "def grade(p, r, g):\n    return 0.0"}, 1],
                    [{"prompt": "criterion"}, 3],
                ],
            },
        )
        with patch.object(
            rm.asyncio, "to_thread", new=AsyncMock(return_value=0)
        ), patch.object(rm, "prompt_score", new=AsyncMock(return_value=1)) as judge:
            self.assertEqual(await rm.custom_rm(None, sample), 0.75)
            self.assertEqual(
                judge.call_args.args[1], sample.metadata["original_prompt"]
            )
            self.assertEqual(judge.call_args.args[2], sample.response)

    async def test_answer_only_reaches_the_judge(self):
        """custom_rm passes each prompt criterion's answer_only to prompt_score."""
        sample = SimpleNamespace(
            response="<think>trace</think>answer",
            label="gold",
            prompt="templated",
            metadata={
                "original_prompt": [{"role": "user", "content": "original"}],
                "rubric": [
                    [{"prompt": "a", "answer_only": True}, 1],
                    [{"prompt": "b"}, 1],
                ],
            },
        )
        with patch.object(rm, "prompt_score", new=AsyncMock(return_value=1)) as judge:
            await rm.custom_rm(None, sample)
        self.assertEqual(
            [c.kwargs["answer_only"] for c in judge.call_args_list], [True, False]
        )

    async def test_answer_only_payload_has_no_thinking(self):
        """With answer_only the judge sees final answers; the prompt stays whole."""
        result = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"score": 1}'))],
        )
        prompt = [{"role": "user", "content": "<think>not a trace</think> task"}]
        response = "<think>long trace</think>\n```cpp\nint main(){}\n```\n"
        gold = "<think>gold trace</think>gold answer"

        async def payload(**kwargs):
            with patch(
                "litellm.acompletion", new=AsyncMock(return_value=result)
            ) as completion:
                await rm.prompt_score("c", prompt, response, gold, **kwargs)
            return json.loads(completion.call_args.kwargs["messages"][1]["content"])

        with patch.dict(
            os.environ,
            {
                "LITELLM_JUDGE_API_KEY": "test-judge",
                "LITELLM_BASE_URL": "https://example.test",
            },
        ):
            sent = await payload(answer_only=True)
            self.assertEqual(sent["response"], "```cpp\nint main(){}\n```")
            self.assertEqual(sent["ground_truth"], "gold answer")
            self.assertEqual(sent["prompt"], prompt)
            default = await payload()
            self.assertEqual(
                (default["response"], default["ground_truth"]), (response, gold)
            )

    def test_final_answer(self):
        """After the last </think>; an unclosed <think> has no answer."""
        self.assertEqual(rm.final_answer("<think>a</think>b</think> c "), "c")
        self.assertEqual(rm.final_answer("<think>cut off mid-trace"), "")
        self.assertEqual(rm.final_answer(" plain "), "plain")
        self.assertEqual(rm.final_answer(None), "")

    async def test_judge_model_and_separate_key(self):
        """Verify judge model and separate key."""
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
            choices=[
                SimpleNamespace(message=SimpleNamespace(content='{"score": 0.75}'))
            ],
        )
        with patch.dict(
            os.environ,
            {
                "LITELLM_JUDGE_API_KEY": "test-judge",
                "LITELLM_API_KEY": "test-generation",
                "LITELLM_BASE_URL": "https://example.test",
            },
        ), patch(
            "litellm.acompletion", new=AsyncMock(return_value=response)
        ) as completion:
            self.assertEqual(
                await rm.prompt_score("criterion", [], "response", "gold"), 0.75
            )
            self.assertEqual(completion.call_args.kwargs["api_key"], "test-judge")
            self.assertEqual(completion.call_args.kwargs["model"], rm.JUDGE_MODEL)
            self.assertEqual(
                completion.call_args.kwargs["reasoning"], {"effort": "low"}
            )

    async def test_consecutive_failure_circuit_and_no_secret_in_error(self):
        """Verify consecutive failure circuit and no secret in error."""
        with patch.dict(
            os.environ,
            {
                "LITELLM_JUDGE_API_KEY": "test-judge",
                "LITELLM_BASE_URL": "https://example.test",
                "JUDGE_MAX_CONSECUTIVE_FAILURES": "3",
            },
        ), patch(
            "litellm.acompletion",
            new=AsyncMock(side_effect=RuntimeError("private provider body")),
        ) as completion, patch.object(
            rm.asyncio, "sleep", new=AsyncMock()
        ):
            with self.assertRaises(rm.JudgeUnavailable) as error:
                await rm.prompt_score("criterion", [], "response", "gold")
            self.assertEqual(completion.await_count, 9)
            self.assertNotIn("private", str(error.exception))
            with self.assertRaises(rm.JudgeUnavailable):
                await rm.prompt_score("criterion", [], "response", "gold")
            self.assertEqual(completion.await_count, 9)

    async def test_circuit_error_names_the_cause_without_secrets(self):
        """The error and one stderr line per streak give class, status, category."""

        class RateLimitError(Exception):
            status_code = 429

        failure = RateLimitError("sk-live-123 private provider body")
        with patch.dict(
            os.environ,
            {
                "LITELLM_JUDGE_API_KEY": "sk-live-123",
                "LITELLM_BASE_URL": "https://example.test",
                "JUDGE_MAX_CONSECUTIVE_FAILURES": "2",
            },
        ), patch(
            "litellm.acompletion", new=AsyncMock(side_effect=failure)
        ), patch.object(
            rm.asyncio, "sleep", new=AsyncMock()
        ), patch(
            "sys.stderr", new_callable=io.StringIO
        ) as stderr:
            with self.assertRaises(rm.JudgeUnavailable) as error:
                await rm.prompt_score("criterion", [], "response", "gold")
        message = str(error.exception)
        self.assertIn("RateLimitError HTTP 429 rate_limit", message)
        self.assertIn("throttling", message)
        lines = stderr.getvalue().splitlines()
        self.assertEqual(len(lines), 1)  # 6 failed attempts, one streak
        self.assertIn("JUDGE_FAILURE", lines[0])
        self.assertIn("RateLimitError HTTP 429 rate_limit", lines[0])
        for text in (message, stderr.getvalue()):
            self.assertNotIn("sk-", text)
            self.assertNotIn("private", text)

    async def test_each_failure_streak_prints_once(self):
        """A success ends the streak, so the next failure prints again."""
        success = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"score": 1}'))],
        )
        not_json = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
            choices=[
                SimpleNamespace(
                    finish_reason="length", message=SimpleNamespace(content="")
                )
            ],
        )
        with patch.dict(
            os.environ,
            {
                "LITELLM_JUDGE_API_KEY": "test",
                "LITELLM_BASE_URL": "https://example.test",
            },
        ), patch(
            "litellm.acompletion",
            new=AsyncMock(
                side_effect=[TimeoutError(), TimeoutError(), success, not_json, success]
            ),
        ), patch.object(
            rm.asyncio, "sleep", new=AsyncMock()
        ), patch(
            "sys.stderr", new_callable=io.StringIO
        ) as stderr:
            self.assertEqual(await rm.prompt_score("c", [], "r", "g"), 1)
            self.assertEqual(await rm.prompt_score("c", [], "r", "g"), 1)
        lines = stderr.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("(TimeoutError timeout)", lines[0])
        self.assertIn("(reply truncated)", lines[1])

    def test_failure_reason_categories(self):
        """Status codes and class names map to categories; messages stay out."""

        class Error(Exception):
            def __init__(self, message, status_code=None):
                super().__init__(message)
                self.status_code = status_code

        cases = [
            (Error("x", 401), "Error HTTP 401 auth"),
            (Error("x", 503), "Error HTTP 503 server_error"),
            (Error("x", 400), "Error HTTP 400 bad_request"),
            (Error("Budget has been exceeded! sk-abc", 400), "Error HTTP 400 budget"),
            (ConnectionError("refused"), "ConnectionError connection"),
            (ValueError("sk-abc"), "ValueError other"),
        ]
        for exc, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(rm.failure_reason(exc=exc), expected)
        reply = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=None)]
        )
        self.assertEqual(rm.failure_reason(result=reply), "reply bad_json")

    async def test_judge_sees_and_instructions_shape_the_request(self):
        """judge_sees limits the payload; instructions extend the system message."""
        result = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"score": 1}'))],
        )
        with patch.dict(
            os.environ,
            {
                "LITELLM_JUDGE_API_KEY": "test",
                "LITELLM_BASE_URL": "https://example.test",
            },
        ), patch(
            "litellm.acompletion", new=AsyncMock(return_value=result)
        ) as completion:
            await rm.prompt_score(
                "mentions apixaban",
                [{"role": "user", "content": "a long pasted note"}],
                "<think>t</think>Start apixaban.",
                "gold",
                answer_only=True,
                judge_sees=("response",),
                instructions="Brand names count.",
            )
            messages = completion.call_args.kwargs["messages"]
            self.assertEqual(
                json.loads(messages[1]["content"]),
                {"criterion": "mentions apixaban", "response": "Start apixaban."},
            )
            self.assertTrue(messages[0]["content"].endswith("Brand names count."))
            await rm.prompt_score("c", [], "r", "g")
            sent = json.loads(completion.call_args.kwargs["messages"][1]["content"])
            self.assertEqual(
                sorted(sent), ["criterion", "ground_truth", "prompt", "response"]
            )
            self.assertNotIn(
                "Instructions", completion.call_args.kwargs["messages"][0]["content"]
            )

    async def test_custom_rm_passes_judge_options(self):
        """Each prompt criterion's judge_sees and instructions reach prompt_score."""
        sample = SimpleNamespace(
            response="answer",
            label="gold",
            prompt="templated",
            metadata={
                "original_prompt": [{"role": "user", "content": "original"}],
                "rubric": [
                    [
                        {
                            "prompt": "a",
                            "judge_sees": ["response"],
                            "instructions": "x",
                        },
                        1,
                    ],
                    [{"prompt": "b"}, 1],
                ],
            },
        )
        with patch.object(rm, "prompt_score", new=AsyncMock(return_value=1)) as judge:
            await rm.custom_rm(None, sample)
        first, second = judge.call_args_list
        self.assertEqual(
            (first.kwargs["judge_sees"], first.kwargs["instructions"]),
            (("response",), "x"),
        )
        self.assertEqual(
            (second.kwargs["judge_sees"], second.kwargs["instructions"]),
            (("prompt", "response", "ground_truth"), ""),
        )

    def test_judge_options_are_validated(self):
        """judge_sees needs "response" and known fields; options are prompt-only."""
        from coding.rl.rubric_data import validate_rubric

        ok = [
            [
                {
                    "prompt": "c",
                    "judge_sees": ["response", "ground_truth"],
                    "instructions": "i",
                },
                1,
            ]
        ]
        self.assertIs(validate_rubric(ok), ok)
        for bad, message in [
            ({"prompt": "c", "judge_sees": ["prompt"]}, "judge_sees"),
            ({"prompt": "c", "judge_sees": ["response", "answer"]}, "judge_sees"),
            ({"prompt": "c", "judge_sees": ["response", "response"]}, "judge_sees"),
            ({"prompt": "c", "instructions": 3}, "instructions"),
            ({"prompt": "c", "instructions": "x" * 4001}, "instructions"),
            (
                {
                    "code": "def grade(p, r, g):\n    return 1.0",
                    "judge_sees": ["response"],
                },
                "applies only",
            ),
            ({"prompt": "c", "other": 1}, "exactly one"),
        ]:
            with self.assertRaisesRegex(ValueError, message):
                validate_rubric([[bad, 1]])

    async def test_content_filter_block_is_unscored_without_retry(self):
        """A content-filter 400 returns None at once and leaves the counter alone."""
        blocked = RuntimeError(
            'Error code: 400 - {"finish_reason":"content_filter","message":'
            '"Response content blocked by label MultiSeverity_SelfHarmScore"}'
        )
        with patch.dict(
            os.environ,
            {
                "LITELLM_JUDGE_API_KEY": "test",
                "LITELLM_BASE_URL": "https://example.test",
                "JUDGE_MAX_CONSECUTIVE_FAILURES": "1",
            },
        ), patch(
            "litellm.acompletion", new=AsyncMock(side_effect=blocked)
        ) as completion, patch.object(
            rm.asyncio, "sleep", new=AsyncMock()
        ):
            for _ in range(3):  # a threshold of 1 would have opened on a failure
                self.assertIsNone(await rm.prompt_score("c", [], "r", "g"))
            self.assertEqual(completion.await_count, 3)

    async def test_content_filter_finish_reason_is_unscored(self):
        """A reply with finish_reason content_filter is unscored, not retried."""
        filtered = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
            choices=[
                SimpleNamespace(
                    finish_reason="content_filter",
                    message=SimpleNamespace(content=""),
                )
            ],
        )
        with patch.dict(
            os.environ,
            {
                "LITELLM_JUDGE_API_KEY": "test",
                "LITELLM_BASE_URL": "https://example.test",
            },
        ), patch(
            "litellm.acompletion", new=AsyncMock(return_value=filtered)
        ) as completion:
            self.assertIsNone(await rm.prompt_score("c", [], "r", "g"))
            self.assertEqual(completion.await_count, 1)

    async def test_blocked_criteria_leave_the_weighted_mean(self):
        """Blocked criteria drop out; an answer with all of them blocked gets 0."""
        sample = SimpleNamespace(
            response="answer",
            label="gold",
            prompt="templated",
            metadata={
                "original_prompt": [{"role": "user", "content": "original"}],
                "rubric": [[{"prompt": "a"}, 1], [{"prompt": "b"}, 3]],
            },
        )
        with patch.object(rm, "prompt_score", new=AsyncMock(side_effect=[0.5, None])):
            self.assertEqual(await rm.custom_rm(None, sample), 0.5)
        with patch.object(rm, "prompt_score", new=AsyncMock(return_value=None)):
            self.assertEqual(await rm.custom_rm(None, sample), 0.0)

    async def test_success_resets_failures(self):
        """Verify success resets failures."""
        success = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"score": 1}'))],
        )
        with patch.dict(
            os.environ,
            {
                "LITELLM_JUDGE_API_KEY": "test",
                "LITELLM_BASE_URL": "https://example.test",
                "JUDGE_MAX_CONSECUTIVE_FAILURES": "2",
            },
        ), patch(
            "litellm.acompletion",
            new=AsyncMock(
                side_effect=[RuntimeError(), success, RuntimeError(), success]
            ),
        ), patch.object(
            rm.asyncio, "sleep", new=AsyncMock()
        ):
            self.assertEqual(await rm.prompt_score("c", [], "r", "g"), 1)
            self.assertEqual(await rm.prompt_score("c", [], "r", "g"), 1)

    async def test_no_generation_key_fallback(self):
        """Verify no generation key fallback."""
        with patch.dict(
            os.environ, {"LITELLM_API_KEY": "generation", "LITELLM_JUDGE_API_KEY": ""}
        ):
            with self.assertRaises(rm.JudgeUnavailable):
                await rm.prompt_score("c", [], "r", "g")


def tinker_checkpoint(directory, sft_sha256):
    """Write a recorded Tinker trial: remote paths plus the SFT receipt."""
    directory.mkdir(parents=True)
    (directory / "tinker_checkpoint.json").write_text(
        json.dumps(
            {
                "base_model": "Qwen/Qwen3-8B",
                "lora_rank": 32,
                "state_path": "tinker://run:train:0/weights/sft",
                "sampler_path": "tinker://run:train:0/sampler_weights/sft",
            }
        )
    )
    (directory / "sft_submission.json").write_text(
        json.dumps({"sft_sha256": sft_sha256})
    )


class FinalizationTests(unittest.TestCase):
    """Regression checks for finalization."""

    def test_freeze_selected_checkpoint_and_detect_tampering(self):
        """Verify freeze selected checkpoint and detect tampering."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            agent = root / "agent"
            agent.mkdir()
            write_rows(agent / "generated_raw.jsonl", rows())
            data.compile_data(agent / "generated_raw.jsonl", root / "prepared")
            checkpoint = agent / "selected"
            tinker_checkpoint(checkpoint, training.digest(root / "prepared/sft.jsonl"))
            (checkpoint / "untrusted.py").write_text("raise Exception()")
            (agent / "submission.json").write_text(
                json.dumps({"checkpoint": str(checkpoint)})
            )
            args = argparse.Namespace(
                data=agent / "generated_raw.jsonl",
                submission=agent / "submission.json",
                agent_dir=agent,
                output_dir=root / "final",
                prepare_only=True,
            )
            compile_without_tokenizer = patch.object(
                training,
                "prepare",
                side_effect=lambda src, dst, model=None: data.compile_data(src, dst),
            )
            with compile_without_tokenizer, patch.dict(
                os.environ, {"TINKER_RL_USAGE_FILE": str(agent / "rl.json")}
            ):
                training.finalize(args)
            frozen = root / "final/initial_checkpoint"
            self.assertEqual(
                sorted(p.name for p in frozen.iterdir()),
                ["sft_submission.json", "tinker_checkpoint.json"],
            )
            command = json.loads((root / "final/final_command.json").read_text())
            self.assertEqual(command[2:4], ["rl.agent_training", "rl-run"])
            self.assertEqual(
                command[command.index("--tinker-usage-file") + 1],
                str(agent / "rl.json"),
            )
            self.assertEqual(command[command.index("--tinker-usd-limit") + 1], "400.0")
            with patch.dict(
                os.environ,
                {
                    "PDB_TASK": "culture",
                    "JUDGE_MODEL": "custom/DeepSeek-V4-Flash-changed",
                    "JUDGE_REASONING_EFFORT": "high",
                    "JUDGE_TOKEN_UNIT_LIMIT": "230000000",
                    "JUDGE_TOKEN_USAGE_FILE": str(root / "different.json"),
                    "TINKER_RL_USAGE_FILE": str(root / "different-rl.json"),
                },
            ), patch.object(training, "run") as run:
                training.run_final(argparse.Namespace(bundle=root / "final"))
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("--task") + 1], "coding")
                self.assertEqual(
                    command[command.index("--judge-token-unit-limit") + 1], "75000000.0"
                )
                self.assertEqual(
                    command[command.index("--judge-token-usage-file") + 1],
                    str(agent / "judge_token_usage.json"),
                )
                self.assertEqual(
                    command[command.index("--judge-reasoning-effort") + 1], "low"
                )
                self.assertEqual(
                    command[command.index("--tinker-usage-file") + 1],
                    str(agent / "rl.json"),
                )
                self.assertNotIn("--skip-final-eval", command)
            (root / "final/data/train.jsonl").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "changed"):
                training.run_final(argparse.Namespace(bundle=root / "final"))

    def test_freeze_rejects_stale_sft_and_foreign_model(self):
        """Reject a receipt for other data and a checkpoint of another base model."""
        with tempfile.TemporaryDirectory() as tmp:
            agent = Path(tmp) / "agent"
            agent.mkdir()
            write_rows(agent / "generated_raw.jsonl", rows())
            tinker_checkpoint(agent / "stale", "0" * 64)
            (agent / "submission.json").write_text(
                json.dumps({"checkpoint": str(agent / "stale")})
            )
            with patch.object(
                training,
                "prepare",
                side_effect=lambda src, dst, model=None: data.compile_data(src, dst),
            ), self.assertRaisesRegex(ValueError, "different"):
                training.freeze(
                    agent / "generated_raw.jsonl",
                    agent / "submission.json",
                    agent,
                    Path(tmp) / "final",
                )
            record = json.loads((agent / "stale/tinker_checkpoint.json").read_text())
            record["base_model"] = "Qwen/Qwen3-4B"
            (agent / "stale/tinker_checkpoint.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "Qwen3-8B"):
                training.checkpoint_files(agent / "stale")

    def test_probe_skips_secret_eval(self):
        """Verify probe skips secret eval and shares the RL ledger."""
        with tempfile.TemporaryDirectory() as tmp:
            ledger = training.TinkerBudget(Path(tmp) / "rl.json", 400, name="rl")
            command = training.rl_command(
                "data", "checkpoint", "out", ledger, probe=True
            )
        self.assertIn("--skip-final-eval", command)
        self.assertEqual(command[command.index("--num-rollout") + 1], "2")
        self.assertEqual(command[command.index("--rollout-batch-size") + 1], "16")


def last_value(command, flag):
    """The wrapper takes the last value given for a flag."""
    return command[len(command) - 1 - command[::-1].index(flag) + 1]


class LocalRLOnlyTests(unittest.TestCase):
    """Local runs are RL-only: base-model start, fixed recipe, no SFT."""

    def test_local_sft_is_refused(self):
        """The sft command names the RL command instead of training."""
        args = argparse.Namespace(backend="local", output_dir="unused")
        with self.assertRaisesRegex(ValueError, "RL-only"):
            training.sft(args)

    def test_recipe_and_default_schedule(self):
        """80 steps x 32 prompts x 8 samples at lr 3e-6; the wrapper sets the length."""
        from coding.rl import local_training

        command = local_training.rl_command("data", "ckpt", "out", "env", "deps")
        expected = {
            "--num-rollout": "80",
            "--rollout-batch-size": "32",
            "--n-samples-per-prompt": "8",
            "--global-batch-size": "256",
            "--lr": "3e-6",
            "--eps-clip-high": "0.28",
            "--weight-decay": "0.1",
            "--adam-beta2": "0.98",
            "--save-interval": "10",
        }
        for flag, value in expected.items():
            with self.subTest(flag=flag):
                self.assertEqual(last_value(command, flag), value)
        # The wrapper picks the response length per task (24,576 for algorithmic).
        self.assertNotIn("--max-response-len", command)
        probe = local_training.rl_command(
            "data", "ckpt", "out", "env", "deps", probe=True
        )
        self.assertEqual(last_value(probe, "--num-rollout"), "2")
        self.assertEqual(last_value(probe, "--save-interval"), "1")

    def test_start_checkpoint_defaults_to_the_base_snapshot(self):
        """No --checkpoint, or the base snapshot itself, starts from base."""
        from coding.rl import local_training

        with tempfile.TemporaryDirectory() as tmp, patch(
            "huggingface_hub.snapshot_download", return_value=tmp
        ):
            base = Path(tmp).resolve()
            self.assertEqual(local_training.start_checkpoint(), base)
            self.assertEqual(local_training.start_checkpoint(tmp), base)
            with self.assertRaises(OSError):
                local_training.start_checkpoint(Path(tmp) / "missing")

    def test_rl_without_checkpoint_runs_the_default_schedule_from_base(self):
        """`rl --data ...` alone: base start, local defaults, lineage recorded."""
        from coding.rl import local_training

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a72"
            write_rows(root / "rows.jsonl", rows())
            args = argparse.Namespace(
                data=str(root / "rows.jsonl"),
                output_dir=str(root / "rl_trial_1"),
                backend="local",
                task="coding",
                checkpoint=None,
                num_rollout=None,
                rollout_batch_size=None,
                n_samples_per_prompt=None,
            )
            commands, starts = [], []
            with patch.object(
                training, "judge_configuration", return_value=None
            ), patch.object(training, "reward_preflight"), patch.object(
                training,
                "prepare",
                side_effect=lambda d, out, model: Path(out),
            ), patch.object(
                training, "record_start", side_effect=lambda d, c: starts.append(c)
            ), patch.object(
                training, "run", side_effect=commands.append
            ), patch.object(
                local_training, "start_checkpoint", return_value=base
            ):
                training.train_rl(args)
        (command,) = commands
        self.assertEqual(starts, [base])
        self.assertEqual(last_value(command, "--checkpoint"), str(base))
        self.assertEqual(last_value(command, "--num-rollout"), "80")
        self.assertEqual(last_value(command, "--rollout-batch-size"), "32")
        self.assertIn("--skip-final-eval", command)


class TinkerBudgetTests(unittest.TestCase):
    """Dollar ledgers shared by agent SFT/RL probes and host RL."""

    def test_prices_cache_hits_and_shared_prompt(self):
        """Bill the shared prompt once, minus cache hits; extra samples are cached."""
        from coding.rl import tinker_budget as budget

        tokens = budget.sample_tokens(1000, 4, 200, [10, 20, 30, 40])
        self.assertEqual(
            tokens, {"prefill": 800, "cached_prefill": 3200, "sample": 100}
        )
        self.assertAlmostEqual(
            budget.cost_usd("Qwen/Qwen3-8B", tokens),
            (800 * 0.195 + 3200 * 0.039 + 100 * 0.60) / 1e6,
        )

    def test_limit_refuses_calls_and_persists_across_instances(self):
        """Refuse a call whose estimate would cross the limit; keep usage on disk."""
        from coding.rl import tinker_budget as budget

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sft.json"
            ledger = budget.TinkerBudget(path, 1.0, name="sft")
            ledger.record({"train": 2_000_000}, session_id="s1")  # $0.88
            again = budget.TinkerBudget(path, 1.0, name="sft")
            self.assertAlmostEqual(again.snapshot()["cost_usd"], 0.88)
            again.check(0.1)
            with self.assertRaises(budget.TinkerBudgetExceeded):
                again.check(0.2)
            self.assertEqual(again.snapshot()["sessions"], ["s1"])
            with self.assertRaisesRegex(ValueError, "limit"):
                budget.TinkerBudget(path, 2.0, name="sft").snapshot()
            unlimited = budget.TinkerBudget(Path(tmp) / "eval.json", None, "eval")
            unlimited.record({"sample": 10**9})
            unlimited.check(10**6)

    def test_env_ledgers_default_limits(self):
        """The launcher's environment selects per-run ledgers and default limits."""
        from coding.rl import tinker_budget as budget

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "TINKER_SFT_USAGE_FILE": str(Path(tmp) / "sft.json"),
                "TINKER_EVAL_USAGE_FILE": str(Path(tmp) / "eval.json"),
            },
        ):
            self.assertEqual(budget.budget_from_env("sft").limit, 150.0)
            self.assertIsNone(budget.budget_from_env("eval").limit)
            with self.assertRaisesRegex(ValueError, "TINKER_RL_USAGE_FILE"):
                budget.budget_from_env("rl")

    def test_sample_requests_respect_concurrency_cap(self):
        """Parallel agents share Tinker's account limit: cap in-flight samples."""
        import asyncio

        from coding.rl import tinker_training as training_module
        from coding.rl.tinker_budget import TinkerBudget

        state = {"active": 0, "peak": 0}

        class Sampler:
            async def sample_async(self, **kwargs):
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
                await asyncio.sleep(0.01)
                state["active"] -= 1
                sequence = SimpleNamespace(tokens=[1, 2], stop_reason="stop")
                return SimpleNamespace(sequences=[sequence], prompt_cache_hit_tokens=0)

        tokenizer = SimpleNamespace(
            apply_chat_template=lambda *args, **kwargs: [5, 6, 7],
            convert_tokens_to_ids=lambda token: 0,
        )

        async def run(budget):
            await asyncio.gather(
                *[
                    training_module.sample_group(
                        Sampler(),
                        tokenizer,
                        [],
                        budget,
                        num_samples=1,
                        max_tokens=4,
                        temperature=1.0,
                    )
                    for _ in range(20)
                ]
            )

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"TINKER_MAX_CONCURRENT_SAMPLES": "2"}
        ):
            budget = TinkerBudget(Path(tmp) / "eval.json", None, "eval")
            asyncio.run(run(budget))
            self.assertEqual(budget.snapshot()["calls"], 20)
        self.assertEqual(state["peak"], 2)

    def test_sft_refuses_a_trial_the_budget_cannot_finish(self):
        """Check the whole trial's cost up front; a cut-off trial saves nothing."""
        from coding.rl import tinker_training as training_module
        from coding.rl.tinker_budget import TinkerBudget, TinkerBudgetExceeded

        tokenizer = SimpleNamespace(
            apply_chat_template=lambda messages, **kwargs: list(
                range(10 if kwargs.get("add_generation_prompt") else 1010)
            )
            + ([] if kwargs.get("add_generation_prompt") else [7, 8]),
            convert_tokens_to_ids=lambda token: 7,
        )
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "sft.jsonl"
            row = {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ]
            }
            data.write_text(json.dumps(row) + "\n")
            budget = TinkerBudget(Path(tmp) / "sft_usage.json", 200, name="sft")
            budget.record({"train": 454_000_000})  # $199.76 spent
            with patch.object(
                training_module, "load_tokenizer", return_value=tokenizer
            ), patch.object(training_module, "service_client") as client:
                with self.assertRaises(TinkerBudgetExceeded):
                    training_module.run_sft(
                        data, Path(tmp) / "trial", budget, epochs=1000
                    )
                client.assert_not_called()

    def test_cosine_schedule(self):
        """Warm up linearly, then decay to zero, as in the HF trainer."""
        from coding.rl.tinker_training import cosine_lr

        self.assertAlmostEqual(cosine_lr(1.0, 0, 100), 1 / 3)
        self.assertAlmostEqual(cosine_lr(1.0, 2, 100), 1.0)
        self.assertLess(cosine_lr(1.0, 99, 100), 0.01)


class RecipeTests(unittest.TestCase):
    """Regression checks for recipe."""

    def test_invalid_judge_config_fails_before_gpu_startup(self):
        """Validate the judge before checking GPU or training dependencies."""
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "data"
            data.mkdir()
            command = [
                "bash",
                str(ROOT / "coding/rl/run_rl_training_and_evaluation.sh"),
                "--rubric-rewards",
                "--rl-data-dir",
                str(data),
                "--judge-model",
                "azure_ai/FW-GLM-5.3",
                "--dry-run",
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("JUDGE_MODEL must contain DeepSeek-V4-Flash", result.stderr)
            self.assertNotIn("nvidia-smi", result.stderr)

    def test_dry_run_builds_fixed_reward_and_valid_shell_without_credentials(self):
        """Verify dry run builds fixed reward and valid shell without credentials."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binaries = root / "bin"
            binaries.mkdir()
            for name, body in {
                "conda": "echo /mock/conda",
                "nvidia-smi": "echo GPU0",
            }.items():
                target = binaries / name
                target.write_text("#!/bin/sh\n" + body + "\n")
                target.chmod(0o755)
            environment = root / "slime"
            (environment / "bin").mkdir(parents=True)
            python = environment / "bin/python"
            python.write_text("#!/bin/sh\nexit 1\n")
            python.chmod(0o755)
            (root / "deps/Megatron-LM").mkdir(parents=True)
            (root / "checkpoint").mkdir()
            (root / "data").mkdir()
            for name in ("train.jsonl", "val.jsonl"):
                (root / "data" / name).write_text("{}\n")
            env = {
                **os.environ,
                "PATH": str(binaries) + ":" + os.environ["PATH"],
                "TVM_FFI_CACHE_DIR": str(root / "tvm"),
                "FLASHINFER_WORKSPACE_BASE": str(root / "flash"),
                "LITELLM_JUDGE_API_KEY": "test-only-secret",
                "SLURM_SUBMIT_DIR": "",
            }
            # The legacy Slime recipe remains runnable directly.
            command = [
                "bash",
                str(ROOT / "coding/rl/run_rl_training_and_evaluation.sh"),
                "--rl-data-dir",
                str(root / "data"),
                "--init-checkpoint",
                str(root / "checkpoint"),
                "--output-dir",
                str(root / "out"),
                "--rubric-rewards",
                "--slime-env",
                str(environment),
                "--slime-deps",
                str(root / "deps"),
                "--dry-run",
            ]
            result = subprocess.run(
                command, env=env, capture_output=True, text=True, check=True
            )
            self.assertNotIn("test-only-secret", result.stdout)
            self.assertIn("--custom-rm-path rl.rubric_rm.custom_rm", result.stdout)
            self.assertIn("enable_thinking", result.stdout)
            self.assertNotIn("--no-thinking", result.stdout)
            start = result.stdout.index("set -euo pipefail\ncd ")
            script = result.stdout[
                start : result.stdout.index("----- (--dry-run:", start)
            ]
            subprocess.run(["bash", "-n"], input=script, text=True, check=True)
            self.assertFalse((root / "out").exists())
            blocked = subprocess.run(
                command + ["--custom-rm-path", "evil.reward"],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(blocked.returncode, 0)


if __name__ == "__main__":
    unittest.main()
