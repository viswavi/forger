"""The culture sandbox copy: opaque ids, no benchmark names, runs without ../coding."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from coding.rl.agent_validation import (
    BENCHMARK_NAMES,
    CULTURE_EVALUATOR_FILES,
    copy_validation_evaluator,
    public_culture_ids,
)
from coding.split_contract import write_manifest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "culture"))
import checkers  # noqa: E402


def row(number, label):
    """A culture task row with an obfuscated gold label."""
    situation = f"Story {number}: Ana brought a gift to her host. Was this acceptable?"
    return {
        "id": 1000 + number,
        "country": "Chile",
        "axis": "Etiquette",
        "subaxis": "gifts",
        "value": "Gratitude",
        "rule_of_thumb": "Bring a small gift.",
        "situation": situation,
        "task_text": f"Country: Chile\nSituation: {situation}",
        "gold_label": checkers.encode_answer(label),
    }


class CultureValidationTests(unittest.TestCase):
    """copy_validation_evaluator(task="culture") on a three-row build."""

    def setUp(self):
        """Write a manifest-bound build with two validation rows and one test row."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name) / "build"
        self.dest = Path(self.tmp.name) / "agent"
        (self.data / ".hidden").mkdir(parents=True)
        self.validation = [row(0, "yes"), row(1, "neutral")]
        test = [row(2, "no")]
        paths = []
        for name, rows in (
            ("validation_tasks.jsonl", self.validation),
            (".hidden/test_tasks.jsonl", test),
        ):
            path = self.data / name
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            paths.append(path)
        write_manifest(self.data, paths, self.validation, test, seed=0, inputs={})
        copy_validation_evaluator(self.data, self.dest, self.data, task="culture")

    def test_copy_has_opaque_ids_and_no_test_rows(self):
        """Ids are val_000..., labels decode, and no file names the benchmark."""
        decoded = [
            json.loads(line)
            for line in (self.dest / "validation_tasks_de_obfuscated.jsonl")
            .read_text()
            .splitlines()
        ]
        self.assertEqual([r["id"] for r in decoded], ["val_000", "val_001"])
        self.assertEqual([r["gold_label"] for r in decoded], ["yes", "neutral"])
        self.assertEqual(
            public_culture_ids(self.data), {1000: "val_000", 1001: "val_001"}
        )
        for path in self.dest.rglob("*"):
            if path.is_file():
                self.assertNotRegex(path.read_text(), BENCHMARK_NAMES)
                self.assertNotIn("Story 2", path.read_text())
        self.assertFalse((self.dest / ".hidden").exists())
        self.assertEqual(
            {p.name for p in self.dest.iterdir()},
            set(CULTURE_EVALUATOR_FILES)
            | {
                "validation_tasks.jsonl",
                "validation_tasks_de_obfuscated.jsonl",
                "validation_manifest.json",
            },
        )

    def test_copy_scores_validation_without_coding(self):
        """The flat copy rescores saved outputs on its own (no ../coding imports)."""
        raw = self.dest / "raw.jsonl"
        raw.write_text(
            json.dumps({"id": "val_000", "output": "<think>x</think>Answer: yes"})
            + "\n"
            + json.dumps({"id": "val_001", "output": "<think>x</think>Answer: no"})
            + "\n"
        )
        env = {**os.environ, "PYTHONPATH": "", "PDB_BENCHMARK_DIR": str(self.dest)}
        out = self.dest / "out"
        subprocess.run(
            [
                sys.executable,
                "evaluate_model.py",
                "--split",
                "val",
                "--from-raw",
                str(raw),
                "--log-dir",
                str(out),
                "--json-output-file",
                str(out / "metrics.json"),
            ],
            cwd=self.dest,
            env=env,
            check=True,
            capture_output=True,
        )
        metrics = json.loads((out / "metrics.json").read_text())
        self.assertAlmostEqual(metrics["context"]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
