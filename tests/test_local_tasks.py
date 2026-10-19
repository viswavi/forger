"""Real local task execution without Docker or a model."""

import pytest

from pyforger.agent_backend import AgentSpec
from pyforger.agent_tasks import TaskSpec, run_command, run_task
from pyforger.local_backend import LocalBackend


@pytest.mark.parametrize("reward", ["1", "1.0"])
def test_absolute_workspace_replay_preserves_data(reward):
    """Absolute workspace replay preserves data."""
    task = TaskSpec(
        instruction="Read /work/data/input.txt and write /work/answer.json",
        files={
            "data/input.txt": b"0.125 /work/data unchanged\x00",
            "analysis.py": (
                "from pathlib import Path\n"
                "Path('/work/answer.json').write_bytes(\n"
                "    Path('/work/data/input.txt').read_bytes())\n"
            ),
        },
        outputs=("answer.json", "analysis.py"),
        verifier=f"test -f /work/answer.json && echo {reward} > /work/reward.txt",
    )
    result = run_command(task, "python3 analysis.py", backend=LocalBackend())
    assert result["passed"] is True
    assert result["files"]["answer.json"] == task.files["data/input.txt"]


@pytest.mark.parametrize(
    "verifier,exit_code,passed",
    [
        ("exit 7", 7, None),
        ("echo 1 > reward.txt; exit 7", 7, None),
        ("true", 0, False),
    ],
)
def test_verifier_cannot_pass_from_failed_or_stale_reward(verifier, exit_code, passed):
    """Broken verification is unscored; stale rewards never produce success."""
    result = run_task(
        TaskSpec("write", files={"reward.txt": "1"}, verifier=verifier),
        agent=AgentSpec(kind="nop"),
        backend=LocalBackend(),
    )
    assert result["passed"] is passed
    assert result["status"] == ("failed" if exit_code else "completed")
    assert result["failure_kind"] == ("infrastructure" if exit_code else None)
    assert result["diagnostics"]["verifier_exit_code"] == exit_code


def test_workspace_mapping_preserves_unrelated_paths(tmp_path):
    """Workspace mapping preserves unrelated paths."""
    from pyforger.local_backend import _workspace_text

    text = (
        "'/work/data' /work/analysis.py /work "
        "https://host/work/data /network/work/data /workspace"
    )
    assert _workspace_text(tmp_path, text) == (
        f"'{tmp_path}/data' {tmp_path}/analysis.py {tmp_path} "
        "https://host/work/data /network/work/data /workspace"
    )


def test_retained_absolute_workspace_script_replays():
    """Retained absolute workspace script replays."""
    from dataclasses import replace

    script = "from pathlib import Path\nPath('/work/answer').write_text('ok')\n"
    task = TaskSpec(
        "run", files={"analysis.py": script}, outputs=("analysis.py", "answer")
    )
    first = run_command(task, "python3 analysis.py", backend=LocalBackend())
    assert first["files"]["analysis.py"] == script.encode()
    second = run_command(
        replace(task, files=first["files"]), "python3 analysis.py", backend=LocalBackend()
    )
    assert second["files"]["answer"] == b"ok"


def test_task_venv_python_used_by_oracle_and_verifier(tmp_path):
    """Task venv python used by oracle and verifier."""
    import subprocess
    import sys

    from pyforger.local_backend import _run_script

    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(tmp_path / ".venv")],
        check=True,
    )
    result = _run_script(tmp_path, "python3 -c 'import sys; print(sys.prefix)'", 10)
    assert result.returncode == 0
    assert result.stdout.strip() == str(tmp_path / ".venv")


def test_oracle_crash_retains_explicit_exit_code():
    """Oracle crash retains explicit exit code."""
    result = run_task(
        TaskSpec("crash"),
        agent=AgentSpec(kind="oracle", solution="exit 7"),
        backend=LocalBackend(),
    )
    assert result["status"] == "failed"
    assert result["diagnostics"]["exit_code"] == 7
    assert result["passed"] is None
