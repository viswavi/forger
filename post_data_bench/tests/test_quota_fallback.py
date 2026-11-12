"""Without a valid submission, the last judge-quota-stopped checkpoint is submitted."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from coding.rl import agent_training


def write_result(run, reason, checkpoint, initial_fallback=False, age=0):
    """An RL run directory whose rl_result.json is `age` seconds old."""
    run.mkdir(parents=True)
    path = run / "rl_result.json"
    path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "termination_reason": reason,
                "initial_checkpoint_fallback": initial_fallback,
            }
        )
    )
    stamp = time.time() - age
    os.utime(path, (stamp, stamp))


class QuotaFallbackTests(unittest.TestCase):
    """freeze_or_quota_fallback and quota_stopped_checkpoints."""

    def setUp(self):
        """An agent directory with completed, quota-stopped and empty runs."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.agent = Path(self.tmp.name).resolve()
        write_result(
            self.agent / "rl_done", "completed", self.agent / "rl_done/hf_rollout79"
        )
        write_result(
            self.agent / "rl_old",
            "judge_quota_exhausted",
            self.agent / "rl_old/hf_rollout29",
            age=600,
        )
        write_result(
            self.agent / "rl_new",
            "judge_quota_exhausted",
            self.agent / "rl_new/hf_rollout59",
            age=60,
        )
        write_result(
            self.agent / "rl_empty",
            "judge_quota_exhausted",
            self.agent / "base",
            initial_fallback=True,
        )

    def test_candidates_are_quota_stopped_runs_newest_first(self):
        """Completed runs and runs that saved no RL checkpoint are skipped."""
        self.assertEqual(
            agent_training.quota_stopped_checkpoints(self.agent),
            [
                str(self.agent / "rl_new/hf_rollout59"),
                str(self.agent / "rl_old/hf_rollout29"),
            ],
        )

    def test_valid_submission_is_kept(self):
        """The agent's own valid selection is never replaced."""
        with patch.object(agent_training, "freeze", return_value=("d", "c")) as freeze:
            result = agent_training.freeze_or_quota_fallback(
                "data", "submission.json", self.agent, "out", "local"
            )
        self.assertEqual(result, ("d", "c", None))
        self.assertEqual(freeze.call_count, 1)

    def test_missing_submission_falls_back_to_newest_quota_run(self):
        """A missing submission.json submits rl_new's last checkpoint."""
        calls = []

        def freeze(data, submission, agent_dir, output, backend, *rest):
            calls.append(Path(submission))
            if len(calls) == 1:
                raise FileNotFoundError("submission.json")
            return ("d", json.loads(Path(submission).read_text())["checkpoint"])

        with patch.object(agent_training, "freeze", side_effect=freeze):
            _, checkpoint, record = agent_training.freeze_or_quota_fallback(
                "data", self.agent / "submission.json", self.agent, "out", "local"
            )
        self.assertEqual(checkpoint, str(self.agent / "rl_new/hf_rollout59"))
        self.assertEqual(record["reason"], "judge_quota_exhausted")
        self.assertIn("FileNotFoundError", record["agent_submission_error"])
        self.assertEqual(calls[1], self.agent / "auto_submission.json")

    def test_invalid_candidate_tries_the_next(self):
        """If the newest candidate fails validation, the older one is used."""

        def freeze(data, submission, agent_dir, output, backend, *rest):
            chosen = Path(submission)
            if chosen.name == "submission.json":
                raise ValueError("has 99 RL steps")
            checkpoint = json.loads(chosen.read_text())["checkpoint"]
            if checkpoint.endswith("hf_rollout59"):
                raise ValueError("prompts outside the final dataset")
            return ("d", checkpoint)

        with patch.object(agent_training, "freeze", side_effect=freeze):
            _, checkpoint, _ = agent_training.freeze_or_quota_fallback(
                "data", self.agent / "submission.json", self.agent, "out", "local"
            )
        self.assertEqual(checkpoint, str(self.agent / "rl_old/hf_rollout29"))

    def test_no_candidate_raises_with_the_original_error(self):
        """No submission and no quota-stopped checkpoint fails as before."""
        with patch.object(
            agent_training, "freeze", side_effect=ValueError("bad selection")
        ), self.assertRaisesRegex(ValueError, "bad selection"):
            agent_training.freeze_or_quota_fallback(
                "data", self.agent / "submission.json", self.agent, "out", "local"
            )


if __name__ == "__main__":
    unittest.main()
