"""Regression tests for rl quota result."""
import contextlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from coding.rl import rubric_rm as rm
from coding.rl.rl_result import select_result

ROOT = Path(__file__).resolve().parents[1]


def checkpoint(path):
    """Create a minimal checkpoint fixture with a valid weights header."""
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "tokenizer_config.json").write_text("{}")
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode()
    (path / "model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + b"1234"
    )
    return path


class QuotaCircuitTests(unittest.IsolatedAsyncioTestCase):
    """Regression checks for quota circuit."""

    async def test_quota_marker_contains_no_provider_body_and_circuit_stays_open(self):
        """Verify quota marker contains no provider body and circuit stays open."""
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "quota.json"
            env = {
                "LITELLM_JUDGE_API_KEY": "test-only",
                "LITELLM_BASE_URL": "https://example.test",
                "JUDGE_MAX_CONSECUTIVE_FAILURES": "2",
                "JUDGE_QUOTA_STOP_FILE": str(marker),
            }
            with patch.dict(os.environ, env), patch(
                "litellm.acompletion",
                new=AsyncMock(
                    side_effect=RuntimeError("Budget has been exceeded! PRIVATE BODY")
                ),
            ) as call, patch.object(rm.asyncio, "sleep", new=AsyncMock()):
                for _ in range(2):
                    with self.assertRaises(rm.JudgeQuotaExhausted):
                        await rm.prompt_score("criterion", [], "answer", "gold")
                self.assertEqual(call.await_count, 6)
                self.assertEqual(
                    json.loads(marker.read_text()), {"reason": "judge_quota_exhausted"}
                )

    async def test_rate_limit_does_not_claim_budget_exhaustion(self):
        """Verify rate limit does not claim budget exhaustion."""
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "quota.json"
            env = {
                "LITELLM_JUDGE_API_KEY": "test-only",
                "LITELLM_BASE_URL": "https://example.test",
                "JUDGE_MAX_CONSECUTIVE_FAILURES": "2",
                "JUDGE_QUOTA_STOP_FILE": str(marker),
            }
            with patch.dict(os.environ, env), patch(
                "litellm.acompletion",
                new=AsyncMock(side_effect=RuntimeError("429 Too many requests")),
            ), patch.object(rm.asyncio, "sleep", new=AsyncMock()):
                with self.assertRaises(rm.JudgeUnavailable) as error:
                    await rm.prompt_score("criterion", [], "answer", "gold")
                self.assertNotIsInstance(error.exception, rm.JudgeQuotaExhausted)
                self.assertFalse(marker.exists())


class ResultTests(unittest.TestCase):
    """Regression checks for result."""

    def setUp(self):
        """Prepare isolated fixtures for this regression check."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.out = self.root / "out"
        self.out.mkdir()
        self.initial = checkpoint(self.root / "initial")
        self.marker = self.root / "quota.json"
        self.marker.write_text('{"reason":"judge_quota_exhausted"}')

    def test_latest_complete_checkpoint_and_warning(self):
        """Verify latest complete checkpoint and warning."""
        checkpoint(self.out / "hf_rollout2")
        latest = checkpoint(self.out / "hf_rollout19")
        partial = checkpoint(self.out / "hf_rollout20")
        (partial / "model.safetensors").write_bytes(b"incomplete")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = select_result(self.out, self.initial, 1, self.marker)
        self.assertEqual(result["checkpoint"], str(latest))
        self.assertEqual(result["termination_reason"], "judge_quota_exhausted")
        self.assertIn("WARNING", stderr.getvalue())
        self.assertEqual(json.loads((self.out / "rl_result.json").read_text()), result)

    def test_before_first_save_returns_initial_with_explicit_warning(self):
        """Verify before first save returns initial with explicit warning."""
        with contextlib.redirect_stderr(io.StringIO()):
            result = select_result(self.out, self.initial, 1, self.marker)
        self.assertEqual(result["checkpoint"], str(self.initial))
        self.assertTrue(result["initial_checkpoint_fallback"])
        self.assertIn("no saved RL progress", result["warning"])

    def test_unrelated_training_failure_remains_fatal(self):
        """Verify unrelated training failure remains fatal."""
        checkpoint(self.out / "hf_rollout19")
        with self.assertRaisesRegex(RuntimeError, "no judge-quota stop"):
            select_result(self.out, self.initial, 7, self.root / "absent")

    def test_success_has_no_warning(self):
        """Verify success has no warning."""
        latest = checkpoint(self.out / "hf_rollout99")
        result = select_result(self.out, self.initial, 0, self.root / "absent")
        self.assertEqual(result["checkpoint"], str(latest))
        self.assertIsNone(result["warning"])
        self.assertEqual(result["termination_reason"], "completed")

    def test_missing_shard_is_not_selected(self):
        """Verify missing shard is not selected."""
        partial = checkpoint(self.out / "hf_rollout19")
        (partial / "model.safetensors.index.json").write_text(
            '{"weight_map":{"a":"missing.safetensors"}}'
        )
        with contextlib.redirect_stderr(io.StringIO()):
            result = select_result(self.out, self.initial, 1, self.marker)
        self.assertTrue(result["initial_checkpoint_fallback"])

    def test_quota_marker_forwarded_to_ray_workers_without_generation_key(self):
        """Verify quota marker forwarded to ray workers without generation key."""
        runtime = self.root / "runtime.json"
        runtime.touch()
        env = {
            "RUBRIC_REWARDS": "1",
            "JUDGE_QUOTA_STOP_FILE": str(self.marker),
            "LITELLM_API_KEY": "test-generation",
            "LITELLM_JUDGE_API_KEY": "test-judge",
        }
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "coding/rl/rubric_runtime_env.py"),
                '{"env_vars":{}}',
                str(runtime),
            ],
            env=env,
            check=True,
        )
        values = json.loads(runtime.read_text())["env_vars"]
        self.assertEqual(values["JUDGE_QUOTA_STOP_FILE"], str(self.marker))
        self.assertEqual(values["LITELLM_JUDGE_API_KEY"], "test-judge")
        self.assertNotIn("LITELLM_API_KEY", values)
        self.assertEqual(runtime.stat().st_mode & 0o777, 0o600)

    def test_actual_shell_tail_continues_to_evaluation_only_for_quota(self):
        """Verify actual shell tail continues to evaluation only for quota."""
        latest = checkpoint(self.out / "hf_rollout19")
        (self.out / "logs").mkdir()
        coding = self.root / "coding"
        (coding / "rl").mkdir(parents=True)
        shutil.copyfile(ROOT / "coding/rl/rl_result.py", coding / "rl/rl_result.py")
        evaluator = coding / "evaluate_model.sh"
        evaluator.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > evaluated.args\n')
        evaluator.chmod(0o755)
        conda = self.root / "conda"
        (conda / "etc/profile.d").mkdir(parents=True)
        (conda / "etc/profile.d/conda.sh").write_text("")
        script = (ROOT / "coding/rl/run_rl_training_and_evaluation.sh").read_text()
        tail = script[script.index("export RUBRIC_REWARDS\n") :]
        common = {
            "OUTPUT_DIR": str(self.out),
            "INIT_CHECKPOINT": str(self.initial),
            "CODING_DIR": str(coding),
            "SLIME_ENV": "/unused",
            "RM_WORKERS": "1",
            "REWARD_DUMP": "/unused",
            "ROLLOUT_LOG_DIR": "/unused",
            "DOUBT_REWARD": "0",
            "DOUBT_MIN": "3",
            "TRAIN_SCRIPT": "mock",
            "SKIP_FINAL_EVAL": "0",
            "NO_THINKING": "0",
            "SPLIT": "test",
            "RUBRIC_REWARDS": "1",
        }
        preamble = (
            "set -euo pipefail\n"
            + "\n".join(k + "=" + shlex.quote(v) for k, v in common.items())
            + "\n"
        )
        preamble += (
            'conda() { if [ "$1" = info ]; then echo '
            + shlex.quote(str(conda))
            + "; fi; }\n"
        )
        for quota in (True, False):
            body = (
                (
                    "printf '%s' '{\"reason\":\"judge_quota_exhausted\"}' > "
                    '"$JUDGE_QUOTA_STOP_FILE"; '
                )
                if quota
                else ""
            )
            command = preamble + "run_in_slime() { " + body + "return 23; }\n" + tail
            result = subprocess.run(
                ["bash"], input=command, text=True, capture_output=True
            )
            if quota:
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("WARNING", result.stderr)
                self.assertIn("FINAL_CHECKPOINT=" + str(latest), result.stdout)
                self.assertIn(str(latest), (coding / "evaluated.args").read_text())
                (coding / "evaluated.args").unlink()
            else:
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((coding / "evaluated.args").exists())


if __name__ == "__main__":
    unittest.main()
