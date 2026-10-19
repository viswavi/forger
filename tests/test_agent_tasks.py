"""Packaged tasks keep explicitly private artifacts out of solver inputs."""

import pytest

from pyforger.agent_tasks import TaskSpec


def test_canonical_directory_separates_private_files(tmp_path):
    """Canonical roundtrips preserve explicit solver and hidden payloads."""
    task = TaskSpec(
        "analyze data",
        files={"data/input.csv": b"1,2"},
        outputs=("answer.json", "analysis.py"),
        verifier_files={
            "grader.py": b"hidden grader",
            "keys/answer.json": b"hidden key",
        },
        solution_files={"analysis.py": b"oracle"},
    )
    root = task.save(tmp_path / "task")
    assert TaskSpec.load(root) == task
    assert not (root / "solver/grader.py").exists()
    assert not (root / "solver/analysis.py").exists()


def test_legacy_directory_is_not_imported(tmp_path):
    """Directories without a canonical manifest are never guessed into tasks."""
    (tmp_path / "instruction.md").write_text("analyze data")
    (tmp_path / "task.toml").write_text("metadata")
    with pytest.raises(FileNotFoundError, match="task.json"):
        TaskSpec.load(tmp_path)


def test_task_roundtrip_through_ancestor_alias(tmp_path):
    """Trusted ancestor aliases preserve the explicit task boundary."""
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    task = TaskSpec("solve", files={"input.txt": b"input"})
    root = task.save(alias / "task")
    assert root == real / "task"
    assert TaskSpec.load(alias / "task") == task
    (alias / "task-link").symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        TaskSpec.load(alias / "task-link")
