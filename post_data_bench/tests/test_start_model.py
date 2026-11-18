"""A release that starts agents from a post-trained checkpoint.

The checkpoint is named by PDB_LOCAL_BASE_MODEL.
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECK = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path
    from coding.rl import backends, lineage, local_training
    from coding.rl.agent_training import check_receipt

    tmp = Path(sys.argv[1])
    out = {"model": backends.MODELS["local"], "base": local_training.BASE_MODEL}
    snapshot = tmp / "hub/models--local--Qwen3-4B-test-rl80/snapshots/abc"
    snapshot.mkdir(parents=True)
    out["snapshot_is_base"] = lineage.is_base_snapshot(snapshot)
    out["stages"] = lineage.stages(snapshot)
    for name, base in (
        ("start", "local/Qwen3-4B-test-rl80"),
        ("qwen", "Qwen/Qwen3-4B"),
    ):
        ckpt = tmp / name
        ckpt.mkdir()
        (ckpt / "rl_submission.json").write_text(json.dumps({
            "stage": "rl",
            "backend": "local",
            "base_model": base,
            "completed_steps": 10,
        }))
        try:
            check_receipt(ckpt, "local")
            out[name] = "accepted"
        except ValueError:
            out[name] = "refused"
    print(json.dumps(out))
    """
)


class StartModelTests(unittest.TestCase):
    """The local student, receipts and lineage follow PDB_LOCAL_BASE_MODEL."""

    def run_check(self, model):
        """Run CHECK with PDB_LOCAL_BASE_MODEL set to model, or unset."""
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        env.pop("PDB_LOCAL_BASE_MODEL", None)
        if model:
            env["PDB_LOCAL_BASE_MODEL"] = model
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "-c", CHECK, tmp],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_starting_model_names_receipts_and_ends_lineage(self):
        """Receipts must name the starting model; its snapshot ends the lineage."""
        out = self.run_check("local/Qwen3-4B-test-rl80")
        self.assertEqual(out["model"], "local/Qwen3-4B-test-rl80")
        self.assertEqual(out["base"], "local/Qwen3-4B-test-rl80")
        self.assertTrue(out["snapshot_is_base"])
        self.assertEqual(out["stages"], [])
        self.assertEqual((out["start"], out["qwen"]), ("accepted", "refused"))

    def test_default_is_qwen3_4b(self):
        """Without the variable, nothing changes: Qwen3-4B receipts are accepted."""
        out = self.run_check(None)
        self.assertEqual(out["model"], "Qwen/Qwen3-4B")
        self.assertEqual((out["start"], out["qwen"]), ("refused", "accepted"))


if __name__ == "__main__":
    unittest.main()
