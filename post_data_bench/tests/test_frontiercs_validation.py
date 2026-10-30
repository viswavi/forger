"""The frontiercs sandbox copy: opaque ids, no benchmark names, no test content."""
import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from coding.rl import decontaminate
from coding.rl.agent_validation import (
    BENCHMARK_NAMES,
    FRONTIERCS_EVALUATOR_FILES,
    copy_validation_evaluator,
)
from coding.split_contract import write_manifest

ROOT = Path(__file__).resolve().parents[1]
NAMES = re.compile(rb"frontier", re.I)
HAS_YAML = importlib.util.find_spec("yaml") is not None
SECRET = "Secret test problem: route the trucks through the warehouse grid"
CONFIG = "type: default\ntime: 1s\nmemory: 256m\nchecker: chk.cc\nsubtasks:\n"
CONFIG += "  - score: 100\n    n_cases: 1\n"


def statement(number):
    """A long statement in the benchmark's style, distinct per problem."""
    words = " ".join(f"token{number}x{k}" for k in range(400))
    return (
        f"Problem {number}. Place the stations {words} so that the total cost is "
        "minimal.\n\nInput\nThe first line contains a single integer n.\n\n"
        "Output\nOutput a single integer."
    )


def task_row(problem_id, kind="default"):
    """A host task row as the split builder writes it."""
    text = SECRET if problem_id == "110" else statement(int(problem_id))
    return {
        "id": problem_id,
        "problem_id": problem_id,
        "type": kind,
        "n_tests": 1,
        "time_limit_s": 1.0,
        "memory_mb": 256,
        "statement": text,
        "task_text": (
            f"Problem:\n\n{text}\n\nJudging limits ...\n\nGenerate solution code:"
        ),
    }


class FrontierValidationTests(unittest.TestCase):
    """copy_validation_evaluator(task="frontiercs") on a three-problem build."""

    def setUp(self):
        """Write a manifest-bound build: validation 26 and 27, secret test 110."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name) / "build"
        self.dest = Path(self.tmp.name) / "agent"
        self.validation = [task_row("26"), task_row("27", "interactive")]
        self.test = [task_row("110")]
        for row, root in (
            (self.validation[0], self.data / "problems"),
            (self.validation[1], self.data / "problems"),
            (self.test[0], self.data / ".hidden/problems"),
        ):
            pdir = root / row["id"]
            (pdir / "testdata").mkdir(parents=True)
            (pdir / "statement.txt").write_text(row["statement"])
            (pdir / "config.yaml").write_text(CONFIG)
            source = "interactor.cc" if row["type"] == "interactive" else "chk.cc"
            (pdir / source).write_text(f"// judge for {row['id']}\nint main(){{}}\n")
            (pdir / "testdata/1.in").write_text("1 2\n")
            (pdir / "testdata/1.ans").write_text("3\n")
        (self.data / ".hidden/problems/110/testdata/1.in").write_text(SECRET)
        (self.data / "testlib").mkdir()
        (self.data / "testlib/testlib.h").write_text("// testlib\n")
        digests = {
            str(path.relative_to(self.data)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for sub in ("problems", ".hidden/problems", "testlib")
            for path in sorted((self.data / sub).rglob("*"))
            if path.is_file()
        }
        (self.data / "problem_files.json").write_text(json.dumps(digests))
        paths = []
        for name, rows in (
            ("validation_tasks.jsonl", self.validation),
            (".hidden/test_tasks.jsonl", self.test),
        ):
            path = self.data / name
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            paths.append(path)
        paths.append(self.data / "problem_files.json")
        write_manifest(self.data, paths, self.validation, self.test, seed=0, inputs={})

    def copy(self):
        """Copy the evaluator into self.dest."""
        copy_validation_evaluator(self.data, self.dest, self.data, task="frontiercs")

    def test_copy_has_promised_files_with_opaque_ids(self):
        """The prompt's files are there, ids are val_000..., test content is not."""
        self.copy()
        files = {
            str(p.relative_to(self.dest)) for p in self.dest.rglob("*") if p.is_file()
        }
        self.assertEqual(
            files,
            set(FRONTIERCS_EVALUATOR_FILES)
            | {
                "validation_tasks.jsonl",
                "validation_manifest.json",
                "testlib/testlib.h",
                "problems/val_000/config.yaml",
                "problems/val_000/chk.cc",
                "problems/val_000/testdata/1.in",
                "problems/val_000/testdata/1.ans",
                "problems/val_001/config.yaml",
                "problems/val_001/interactor.cc",
                "problems/val_001/testdata/1.in",
                "problems/val_001/testdata/1.ans",
            },
        )
        rows = [
            json.loads(line)
            for line in (self.dest / "validation_tasks.jsonl").read_text().splitlines()
        ]
        self.assertEqual([r["id"] for r in rows], ["val_000", "val_001"])
        self.assertEqual(
            [list(r) for r in rows],
            [
                [
                    "id",
                    "type",
                    "n_tests",
                    "time_limit_s",
                    "memory_mb",
                    "statement",
                    "task_text",
                ]
            ]
            * 2,
        )
        self.assertEqual(rows[1]["statement"], statement(27))
        for path in self.dest.rglob("*"):
            self.assertNotRegex(str(path.relative_to(self.dest)), BENCHMARK_NAMES)
            if path.is_file():
                data = path.read_bytes()
                self.assertIsNone(NAMES.search(data), path)
                self.assertNotIn(SECRET.encode(), data)
                self.assertNotIn(b"judge for 110", data)
                if path.suffix in (".py", ".sh"):
                    self.assertNotRegex(data.decode(), BENCHMARK_NAMES)

    def test_changed_problem_file_is_refused(self):
        """A validation file that differs from problem_files.json is not copied."""
        (self.data / "problems/26/testdata/1.ans").write_text("4\n")
        with self.assertRaisesRegex(ValueError, "differs from problem_files"):
            self.copy()

    @unittest.skipUnless(HAS_YAML, "the judge reads config.yaml with PyYAML")
    def test_copy_scores_validation_without_coding(self):
        """The flat copy verifies its manifest and rescores outputs on its own."""
        self.copy()
        raw = self.dest / "raw.jsonl"
        raw.write_text(
            "".join(
                json.dumps({"id": f"val_00{k}", "output": "<think>reasoning cut off"})
                + "\n"
                for k in range(2)
            )
        )
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
            ],
            cwd=self.dest,
            env={**os.environ, "PYTHONPATH": "", "PDB_BENCHMARK_DIR": str(self.dest)},
            check=True,
            capture_output=True,
        )
        metrics = json.loads((out / "metrics.json").read_text())
        self.assertEqual((metrics["n_total"], metrics["n_no_code"]), (2, 2))
        predictions = (out / "predictions.jsonl").read_text().splitlines()
        self.assertEqual(
            [json.loads(p)["id"] for p in predictions], ["val_000", "val_001"]
        )

    def test_from_raw_rejudges_every_saved_sample(self):
        """A problem scores the mean of its samples; metrics count each sample."""
        self.copy()
        raw = self.dest / "raw.jsonl"
        sample = {"output": "<think>reasoning cut off", "finish_reason": "length"}
        raw.write_text(
            "".join(
                json.dumps({"id": f"val_00{k}", "samples": [sample, sample, sample]})
                + "\n"
                for k in range(2)
            )
        )
        out = self.dest / "out_samples"
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
            ],
            cwd=self.dest,
            env={**os.environ, "PYTHONPATH": "", "PDB_BENCHMARK_DIR": str(self.dest)},
            check=True,
            capture_output=True,
        )
        metrics = json.loads((out / "metrics.json").read_text())
        self.assertEqual(
            (metrics["n_total"], metrics["n_samples"], metrics["n_truncated"]),
            (2, 6, 6),
        )
        row = json.loads((out / "predictions.jsonl").read_text().splitlines()[0])
        self.assertEqual((row["n_samples"], row["score"]), (3, 0.0))

    @unittest.skipUnless(
        shutil.which("g++") and sys.platform.startswith("linux") and HAS_YAML,
        "needs g++ on Linux and PyYAML",
    )
    def test_copy_judges_programs_from_a_read_only_mount(self):
        """Hosts mount the benchmark read-only: compiled judges go elsewhere."""
        self.copy()
        raw = Path(self.tmp.name) / "raw.jsonl"
        program = '#include <cstdio>\nint main(){puts("3");}'
        raw.write_text(
            json.dumps(
                {"id": "val_000", "output": f"<think>r</think>```cpp\n{program}\n```"}
            )
            + "\n"
        )
        out = Path(self.tmp.name) / "out"
        subprocess.run(["chmod", "-R", "a-w", str(self.dest)], check=True)
        self.addCleanup(subprocess.run, ["chmod", "-R", "u+w", str(self.dest)])
        env = {
            **os.environ,
            "PYTHONPATH": "",
            "PDB_BENCHMARK_DIR": str(self.dest),
            "CPP_JUDGE_CACHE_DIR": str(Path(self.tmp.name) / "cache"),
        }
        (Path(self.tmp.name) / "cache").mkdir(mode=0o700)
        result = subprocess.run(
            [
                sys.executable,
                "evaluate_model.py",
                "--split",
                "val",
                "--from-raw",
                str(raw),
                "--log-dir",
                str(out),
                "--limit",
                "1",
            ],
            cwd=self.dest,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        row = json.loads((out / "predictions.jsonl").read_text().splitlines()[0])
        self.assertEqual(row["score"], 100.0)
        self.assertEqual([s["status"] for s in row["samples"]], ["done"])

    def test_launcher_remaps_baseline_and_writes_signatures(self):
        """Baseline rows get val ids without problem_id; signatures hold no text."""
        tree = ast.parse((ROOT / "launch_agent.py").read_text())
        selected = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"_copy_into", "_public_source", "_task_public_copies"}
        ]
        namespace = dict(
            os=os,
            shutil=shutil,
            args=SimpleNamespace(
                training_mode="sft-rl",
                task="frontiercs",
                backend="local",
                model="Qwen/Qwen3-4B",
            ),
            TASK_PUBLIC_FILES=[],
            VAL_OUTPUT_FILES=("predictions.jsonl", "metrics.json"),
            SHARED_TASK_DIR=str(ROOT / "coding"),
        )
        exec(
            compile(
                ast.Module(body=selected, type_ignores=[]), "launch_agent.py", "exec"
            ),
            namespace,
        )
        from coding.eval_state import CONTRACT_VERSION
        from coding.split_contract import sha256

        val = self.data / "model_outputs/Qwen__Qwen3-4B/val"
        val.mkdir(parents=True)
        (val / "predictions.jsonl").write_text(
            "".join(
                json.dumps({"id": i, "problem_id": i, "score": 1.0}) + "\n"
                for i in ("27", "26")
            )
        )
        (val / "metrics.json").write_text(
            json.dumps(
                {
                    "evaluation_contract": CONTRACT_VERSION,
                    "tasks_sha256": sha256(self.data / "validation_tasks.jsonl"),
                    "source": "/data/frontiercs/raw.jsonl",
                }
            )
        )
        namespace["_task_public_copies"](str(self.data), str(self.dest))
        copied = self.dest / "model_outputs/Qwen__Qwen3-4B/val"
        self.assertEqual(
            [json.loads(line) for line in (copied / "predictions.jsonl").open()],
            [{"id": "val_001", "score": 1.0}, {"id": "val_000", "score": 1.0}],
        )
        self.assertNotIn("source", json.loads((copied / "metrics.json").read_text()))
        signature = json.loads(
            (self.dest / "rl" / decontaminate.SIGNATURES.name).read_text()
        )
        self.assertEqual(
            (signature["n"], signature["fraction"]),
            (decontaminate.N, decontaminate.LONG_FRACTION),
        )
        self.assertNotIn("Secret", json.dumps(signature))
        self.assertTrue((self.dest / "problems/val_000/chk.cc").is_file())


class FrontierDecontaminationTests(unittest.TestCase):
    """Long statements: a copied statement is flagged, generic phrasing is not."""

    def setUp(self):
        """Signatures of two long test statements, as the launcher builds them."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root / ".hidden").mkdir()
        write = lambda path, rows: path.write_text(  # noqa: E731
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        self.test = [task_row("3"), task_row("4")]
        write(root / ".hidden/test_tasks.jsonl", self.test)
        write(root / "validation_tasks.jsonl", [task_row("5")])
        self.signature = decontaminate.benchmark_signatures(root)

    def flagged(self, text):
        """Whether a row whose user turn is `text` overlaps a test statement."""
        row = {
            "prompt": [
                {"role": "system", "content": "You are a competitive programmer."},
                {"role": "user", "content": text},
            ]
        }
        return decontaminate.overlaps(row, self.signature)

    def test_copied_statement_is_flagged(self):
        """The whole statement, or a fifth of it, inside a prompt is flagged."""
        self.assertTrue(self.flagged(self.test[0]["task_text"]))
        words = self.test[1]["statement"].split()
        start = len(words) // 3
        self.assertTrue(
            self.flagged("Solve: " + " ".join(words[start : start + len(words) // 5]))
        )

    def test_generic_statement_is_not_flagged(self):
        """Standard input/output phrasing shared with the test set does not count."""
        generic = (
            "Problem:\n\nYou are given a tree with n vertices. Find the diameter.\n\n"
            "Input\nThe first line contains a single integer n. Each of the next "
            "n - 1 lines contains two integers u and v.\n\nOutput\nOutput a single "
            "integer.\n\nGenerate solution code:"
        )
        self.assertFalse(self.flagged(generic))


if __name__ == "__main__":
    unittest.main()
