"""Launcher budget persistence and hidden-library accounting exports."""

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from coding.rl.token_budget import DEFAULT_JUDGE_MODEL, TokenUnitBudget

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def launcher_functions(*names, **namespace):
    """Load helpers without starting the launcher's executable workflow."""
    tree = ast.parse((ROOT / "launch_agent.py").read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    scope = {"os": os, "shutil": shutil, **namespace}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "launch_agent.py", "exec"),
        scope,
    )
    return scope


class AgentTokenBudgetTests(unittest.TestCase):
    """Check persistent budgets and sandbox helper exports."""

    def test_recovery_retains_both_ledgers_and_judge_configuration(self):
        """Preserve usage and judge settings across terminal sessions."""
        configure = launcher_functions("configure_run_token_budgets")[
            "configure_run_token_budgets"
        ]
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {}, clear=True
        ):
            os.environ["JUDGE_MODEL"] = "custom/DEEPSEEK-V4-FLASH-test"
            os.environ["JUDGE_REASONING_EFFORT"] = "high"
            configure(directory, "coding")
            generation = TokenUnitBudget(
                os.environ["FORGER_TOKEN_USAGE_FILE"], 230_000_000
            )
            judge = TokenUnitBudget(os.environ["JUDGE_TOKEN_USAGE_FILE"], 75_000_000)
            generation.record(DEFAULT_JUDGE_MODEL, 10, 20)
            judge.record(DEFAULT_JUDGE_MODEL, 5, 3)
            os.environ.clear()  # Recovery starts in a fresh terminal session.
            configure(directory, "coding")
            self.assertEqual(generation.snapshot()["token_units"], 70)
            self.assertEqual(judge.snapshot()["token_units"], 14)
            self.assertEqual(os.environ["JUDGE_MODEL"], "custom/DEEPSEEK-V4-FLASH-test")
            self.assertEqual(os.environ["JUDGE_REASONING_EFFORT"], "high")
            self.assertEqual(os.environ["PDB_TASK"], "coding")

    def test_recovery_rejects_changed_task_without_touching_ledgers(self):
        """Reject a mismatched task before modifying saved usage."""
        configure = launcher_functions("configure_run_token_budgets")[
            "configure_run_token_budgets"
        ]
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {}, clear=True
        ):
            configure(directory, "coding")
            generation = Path(os.environ["FORGER_TOKEN_USAGE_FILE"])
            judge = Path(os.environ["JUDGE_TOKEN_USAGE_FILE"])
            original = generation.read_bytes(), judge.read_bytes()
            with self.assertRaisesRegex(ValueError, "task or policy"):
                configure(directory, "culture")
            self.assertEqual((generation.read_bytes(), judge.read_bytes()), original)

    def test_saved_configuration_contains_no_credentials_or_external_paths(self):
        """Persist only public policy and keep usage inside the run directory."""
        configure = launcher_functions("configure_run_token_budgets")[
            "configure_run_token_budgets"
        ]
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "LITELLM_API_KEY": "generation-secret",
                "LITELLM_JUDGE_API_KEY": "judge-secret",
                "JUDGE_TOKEN_USAGE_FILE": "/outside/run/usage.json",
            },
            clear=True,
        ):
            configure(directory, "coding")
            config = json.loads(
                (Path(directory) / "token_budget_config.json").read_text()
            )
            self.assertEqual(
                config,
                {
                    "version": 1,
                    "task": "coding",
                    "generation_token_unit_limit": 230_000_000,
                    "judge_token_unit_limit": 75_000_000,
                    "judge_model": DEFAULT_JUDGE_MODEL,
                    "judge_reasoning_effort": "low",
                },
            )
            self.assertEqual(
                Path(os.environ["JUDGE_TOKEN_USAGE_FILE"]).parent, Path(directory)
            )

    def test_recovery_refuses_symlinked_configuration(self):
        """Reject a configuration symlink without reading or modifying its target."""
        configure = launcher_functions("configure_run_token_budgets")[
            "configure_run_token_budgets"
        ]
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {}, clear=True
        ):
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            target = root / "outside.json"
            target.write_text("unchanged")
            (run / "token_budget_config.json").symlink_to(target)
            with self.assertRaises(OSError):
                configure(run, "coding")
            self.assertEqual(target.read_text(), "unchanged")
            self.assertFalse((run / "generation_token_usage.json").exists())

    def test_new_run_uses_its_own_ledger(self):
        """Isolate usage between separate agent runs."""
        configure = launcher_functions("configure_run_token_budgets")[
            "configure_run_token_budgets"
        ]
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {}, clear=True
        ):
            first, second = Path(directory) / "first", Path(directory) / "second"
            configure(first, "culture")
            previous = TokenUnitBudget(
                os.environ["FORGER_TOKEN_USAGE_FILE"], 230_000_000
            )
            previous.record(DEFAULT_JUDGE_MODEL, 1, 1)
            configure(second, "culture")
            self.assertEqual(Path(os.environ["FORGER_TOKEN_USAGE_FILE"]).parent, second)
            self.assertEqual(
                TokenUnitBudget(
                    os.environ["FORGER_TOKEN_USAGE_FILE"], 230_000_000
                ).snapshot()["token_units"],
                0,
            )
            self.assertEqual(previous.snapshot()["token_units"], 4)
            self.assertEqual(os.environ["JUDGE_TOKEN_UNIT_LIMIT"], "75000000")

    def test_invalid_judge_configuration_fails_before_creating_run(self):
        """Validate judge settings before creating any run files."""
        configure = launcher_functions("configure_run_token_budgets")[
            "configure_run_token_budgets"
        ]
        for values in (
            {"JUDGE_MODEL": "custom/GLM-5.3"},
            {"JUDGE_REASONING_EFFORT": "medium"},
        ):
            with self.subTest(
                values=values
            ), tempfile.TemporaryDirectory() as directory:
                run = Path(directory) / "run"
                with patch.dict(os.environ, values, clear=True), self.assertRaises(
                    ValueError
                ):
                    configure(run, "coding")
                self.assertFalse(run.exists())

    def test_hidden_forger_exports_working_helpers_for_each_task(self):
        """Meter sandbox calls with the Forger library hidden."""
        scope = launcher_functions(
            "build_agent_mounts",
            "_copy_into",
            "_touch_placeholder",
            args=SimpleNamespace(training_mode="sft"),
            _task_public_copies=lambda source, target: None,
        )
        for task in ("coding", "culture"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                repo = root / "source"
                for relative in (
                    "post_data_bench/coding/rl/token_budget.py",
                    "src/pyforger/token_units.py",
                ):
                    destination = repo / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(REPO / relative, destination)
                task_dir = repo / "post_data_bench" / task
                task_dir.mkdir(parents=True, exist_ok=True)
                scratch = root / "copy"
                run = root / "run"
                run.mkdir()
                scope["build_agent_mounts"](
                    forger_repo=str(repo),
                    curator_repo=str(root / "curator"),
                    task_dir=str(task_dir),
                    agent_data_dir=str(run),
                    scratch_dir=str(scratch),
                    hide_forger=True,
                    hide_curator=True,
                )
                exported_task = scratch / "forger/post_data_bench" / task
                self.assertFalse((scratch / "forger/src").exists())
                # Neither the repository nor pyforger is on the isolated sys.path.
                code = """
import sys
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
sys.modules['litellm'] = SimpleNamespace(completion=lambda **kwargs: {
    'usage': {'prompt_tokens': 10, 'completion_tokens': 4}})
from rl.token_budget import completion
completion(model='litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731')
"""
                ledger = run / "generation_token_usage.json"
                result = subprocess.run(
                    [sys.executable, "-I", "-c", code, str(exported_task)],
                    env={
                        "FORGER_TOKEN_USAGE_FILE": str(ledger),
                        "FORGER_TOKEN_UNIT_LIMIT": "100",
                    },
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(ledger.read_text())["token_units"], 22)

    def test_prompts_format_with_token_units(self):
        """Render each supplied prompt with token quotas and accounting helpers."""
        values = dict(
            TASK_DIR="/task",
            STUDENT_MODEL="Qwen/Qwen3-4B",
            MAX_SFT_LENGTH="18800",
            GENERATION_TOKEN_UNIT_LIMIT="230000000",
            GENERATION_USAGE_PATH="/run/generation_token_usage.json",
            TIMEOUT_HOURS="6",
            GPU_DESCRIPTION="assigned GPUs",
            AGENT_DATA_DIR="/run",
            FORGER_REPO="/forger",
        )
        for relative in (
            "coding/FORGER_PROMPT.txt",
            "culture/FORGER_PROMPT.txt",
            "coding/minimal_scaffold_prompt.txt",
        ):
            with self.subTest(prompt=relative):
                prompt = (ROOT / relative).read_text().format(**values)
                self.assertIn("230000000 token units", prompt)
                self.assertIn("from rl.token_budget import completion", prompt)
                self.assertNotIn("llm_spend", prompt)


if __name__ == "__main__":
    unittest.main()
