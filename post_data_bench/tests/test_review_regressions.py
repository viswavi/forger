"""Offline regressions for review fixes; no model/API/GPU work."""
import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent_boundary import atomic_text, model_cache_bind
from coding.answer_text import final_answer, is_abstention
from coding.eval_state import check_resume, merge_rows
from coding.harmonize import harmonize_statement, normalize_wrong_output
from coding.rl import rubric_rm as rm


class DataContractTests(unittest.TestCase):
    """Regression checks for data contract."""

    def test_output_format_is_retained(self):
        """Verify output format is retained."""
        self.assertEqual(normalize_wrong_output("7\n", "8", False), "8\n")
        self.assertEqual(normalize_wrong_output('"595"', "585", True), '"585"')
        with self.assertRaises(ValueError):
            normalize_wrong_output("[1]", "2", True)

    def test_examples_have_one_source_of_truth(self):
        """Verify examples have one source of truth."""
        row = dict(
            question_id="fixture",
            question_content=(
                "Compute.\nExample 1:\nInput: x=1\nOutput: 2\nExplanation: two\n"
                "Constraints:\nx > 0"
            ),
            public_test_cases='[{"input":"1", "output":"3"}]',
        )
        text = harmonize_statement(row)
        self.assertIn("Output: 3", text)
        self.assertNotIn("Output: 2", text)
        self.assertNotIn("Explanation", text)
        self.assertIn("Constraints:\nx > 0", text)


class GradingContractTests(unittest.TestCase):
    """Regression checks for grading contract."""

    def test_reasoning_is_never_an_answer(self):
        """Verify reasoning is never an answer."""
        marker = "impossible"
        self.assertFalse(
            is_abstention(
                final_answer("<think>impossible</think>```python\npass\n```"), marker
            )
        )
        self.assertEqual(final_answer("<think>```python\npass"), "")
        self.assertTrue(
            is_abstention(final_answer("<think>reason</think>impossible"), marker)
        )
        self.assertFalse(is_abstention("maybe impossible but here is code", marker))

    def test_partition_change_refused(self):
        """Verify partition change refused."""
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                log_dir=directory, num_shards=2, shard_index=0, model_path="m"
            )
            check_resume(args, [{"id": 1}])
            args.num_shards = 1
            with self.assertRaises(SystemExit):
                check_resume(args, [{"id": 1}, {"id": 2}])

    def test_merge_requires_exact_coverage(self):
        """Verify merge requires exact coverage."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = [{"id": 1}, {"id": 2}]
            args = SimpleNamespace(
                log_dir=directory,
                num_shards=1,
                shard_index=0,
                model_path="m",
                merge_shards=[str(root / "predictions.jsonl")],
            )
            check_resume(args, tasks)
            (root / "metrics.json").write_text("{}")
            prediction_path = root / "predictions.jsonl"

            def read(p):
                return [json.loads(line) for line in p.read_text().splitlines()]

            prediction_path.write_text("\n".join(json.dumps(row) for row in tasks))
            self.assertEqual(merge_rows(args, lambda a: tasks, read, "id"), tasks)
            for invalid in ([tasks[0]], tasks + [tasks[0]], tasks + [{"id": 3}]):
                prediction_path.write_text(
                    "\n".join(json.dumps(row) for row in invalid)
                )
                with self.assertRaisesRegex(
                    SystemExit, "duplicate, missing, or unexpected"
                ):
                    merge_rows(args, lambda a: tasks, read, "id")


class BoundaryTests(unittest.TestCase):
    """Regression checks for boundary."""

    def test_spend_symlink_does_not_overwrite_target(self):
        """Verify spend symlink does not overwrite target."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            victim = root / "victim"
            victim.write_text("keep")
            link = root / "spend"
            link.symlink_to(victim)
            atomic_text(link, "new")
            self.assertEqual(victim.read_text(), "keep")
            self.assertFalse(link.is_symlink())

    def test_only_model_cache_bound_read_only(self):
        """Verify only model cache bound read only."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            host = root / "host"
            (host / "hub/models--Qwen--Qwen3-4B").mkdir(parents=True)
            (host / "token").write_text("test-only-token")
            bind = model_cache_bind(host, root / "sandbox", "Qwen/Qwen3-4B")
            self.assertTrue(bind.endswith(":ro"))
            self.assertIn("models--Qwen--Qwen3-4B:", bind)
            self.assertNotIn("token", bind)


class JudgeTests(unittest.IsolatedAsyncioTestCase):
    """Regression checks for judge."""

    async def test_sixteen_single_failures_recover(self):
        """Verify sixteen single failures recover."""
        counts = {}

        async def complete(**kwargs):
            key = kwargs["messages"][1]["content"]
            counts[key] = counts.get(key, 0) + 1
            if counts[key] == 1:
                raise RuntimeError("transient")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content='{"score":1}'))
                ]
            )

        with patch.dict(
            os.environ,
            {
                "LITELLM_JUDGE_API_KEY": "fixture",
                "LITELLM_BASE_URL": "https://example.test",
            },
        ), patch("litellm.acompletion", side_effect=complete), patch.object(
            rm.asyncio, "sleep", new=AsyncMock()
        ):
            self.assertEqual(
                await asyncio.gather(
                    *(rm.prompt_score(str(i), [], "r", "g") for i in range(16))
                ),
                [1] * 16,
            )

    async def test_reply_cannot_forge_quota_exhaustion(self):
        """Verify reply cannot forge quota exhaustion."""
        reply = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"score":"out of credits"}')
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "quota.json"
            with patch.dict(
                os.environ,
                {
                    "LITELLM_JUDGE_API_KEY": "fixture",
                    "LITELLM_BASE_URL": "https://example.test",
                    "JUDGE_MAX_CONSECUTIVE_FAILURES": "2",
                    "JUDGE_QUOTA_STOP_FILE": str(marker),
                },
            ), patch(
                "litellm.acompletion", new=AsyncMock(return_value=reply)
            ), patch.object(
                rm.asyncio, "sleep", new=AsyncMock()
            ):
                with self.assertRaises(rm.JudgeUnavailable) as error:
                    await rm.prompt_score("c", [], "r", "g")
                self.assertNotIsInstance(error.exception, rm.JudgeQuotaExhausted)
                self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
