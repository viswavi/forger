"""plot_training finds a run's logs from its job log and plots them."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coding"))

from rl.plot_training import plot, resolve, summary  # noqa: E402

TAG = "20261019_204434_c9fc95"


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _run_tree() -> tuple[Path, Path]:
    """A models dir with one run, and the job log that names it."""
    base = Path(tempfile.mkdtemp())
    models = base / "agent_data_models" / "coding"
    trial = models / "agent_runs" / f"agent_run_{TAG}" / "sft_trial_4"
    _jsonl(
        trial / "train_log.jsonl", [{"step": 1, "loss": 1.3}, {"step": 2, "loss": 0.9}]
    )
    final = models / f"Qwen__Qwen3-8B_{TAG}_final"
    (final / "rl").mkdir(parents=True)
    (final / "submission_manifest.json").write_text(
        json.dumps({"selected_checkpoint": str(trial)})
    )
    _jsonl(final / "data" / "train.jsonl", [{"prompt": []}] * 3)
    _jsonl(
        final / "rl" / "train_log.jsonl",
        [
            {
                "step": 1,
                "reward_mean": 0.5,
                "groups_with_signal": 64,
                "response_len_mean": 900,
                "truncated_frac": 0.1,
            },
            {
                "step": 2,
                "reward_mean": 0.8,
                "groups_with_signal": 32,
                "response_len_mean": 800,
                "truncated_frac": 0.0,
            },
        ],
    )
    _jsonl(
        final / "rl" / "heldout_log.jsonl", [{"step": 2, "heldout_reward_mean": 0.7}]
    )
    _jsonl(
        final / "rl" / "rollouts.jsonl",
        [
            {"step": 1, "rewards": [1.0, 1.0]},
            {"step": 1, "rewards": [0.0, 1.0]},
            {"step": 2, "rewards": [0.0, 0.0]},
        ],
    )
    log = base / "runs" / "pdb-native-v4_10860388.out"
    log.parent.mkdir()
    log.write_text(
        f"cd {models}/agent_runs/agent_run_{TAG} && python x.py\n"
        f"--data {models}/agent_runs/agent_run_{TAG}/generated_raw.jsonl\n"
    )
    return models, log


class PlotTrainingTests(unittest.TestCase):
    """From a job log to the summary line and the figure."""

    def test_job_log_finds_the_run_and_its_logs(self):
        """The log's run tag and path locate the final dir and the submitted trial."""
        models, log = _run_tree()
        run = resolve(str(log), None)
        self.assertEqual(run.label, "pdb-native-v4")
        self.assertEqual(run.final_dir, models / f"Qwen__Qwen3-8B_{TAG}_final")
        self.assertEqual([r["loss"] for r in run.sft()], [1.3, 0.9])
        self.assertEqual(
            run.rollouts_by_step(),
            {
                1: {"solved": 0.5, "failed": 0.0, "flat": 0.5},
                2: {"solved": 0.0, "failed": 1.0, "flat": 1.0},
            },
        )
        self.assertIn("rows     3", summary(run))
        self.assertIn("RL reward 0.50→0.80", summary(run))
        self.assertIn("all solved 50%→0%", summary(run))

    def test_final_dir_and_figure(self):
        """A final dir works as an argument too, and the figure is written."""
        models, _ = _run_tree()
        run = resolve(str(models / f"Qwen__Qwen3-8B_{TAG}_final"), None)
        out = models / "training.png"
        plot([run], out)
        self.assertGreater(out.stat().st_size, 10_000)

    def test_missing_logs_leave_panels_empty(self):
        """A run without RL logs still plots and summarizes."""
        models = Path(tempfile.mkdtemp())
        run = resolve(str(models / f"Qwen__Qwen3-8B_{TAG}_final"), None)
        self.assertIn("RL reward -", summary(run))
        plot([run], models / "empty.png")


if __name__ == "__main__":
    unittest.main()
