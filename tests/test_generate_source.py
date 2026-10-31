"""Source generation retries real program failures, not backend execution errors."""

from __future__ import annotations

import shlex

import pytest

from pyforger import agents
from pyforger.agent_backend import AgentSpec
from pyforger.local_backend import LocalBackend
from pyforger.structured import ValidationExhausted


class RecordingLocalBackend(LocalBackend):
    """Record prompts while running real oracle scripts and restricted replay."""

    def __init__(self):
        """Initialize the observed prompts."""
        self.instructions = []

    def run_agent(self, task, agent):
        """Record each attempt without replacing execution."""
        self.instructions.append(task.instruction)
        return super().run_agent(task, agent)


def test_source_repairs_replay_failure(tmp_path):
    """An invalid return type yields feedback and a new valid program."""
    marker = shlex.quote(str(tmp_path / "first-attempt"))
    agent = AgentSpec(
        kind="oracle",
        solution=(
            f"if test -f {marker}; then\n"
            "printf 'def run(item):\n"
            '    return [item["text"].upper()]\n\' > program.py\n'
            "else\n"
            f"touch {marker}\n"
            "printf 'def run(item):\n    return 42\n' > program.py\n"
            "fi"
        ),
    )
    backend = RecordingLocalBackend()
    code = agents.generate_source(
        "uppercase", {"text": "abc"}, agent=agent, backend=backend
    )
    assert backend.execute_program(code, {"text": "xyz"}, 10) == ["XYZ"]
    assert len(backend.instructions) == 2
    assert "list of strings" in backend.instructions[1]
    assert "previous program failed validation" in backend.instructions[1]


def test_source_exhaustion_retains_rejected_program():
    """Exhaustion preserves the legacy message and structured cause/history."""
    backend = RecordingLocalBackend()
    agent = AgentSpec(
        kind="oracle", solution="echo 'def run(item): return 42' > program.py"
    )
    with pytest.raises(
        RuntimeError, match="did not produce a working program"
    ) as caught:
        agents.generate_source("goal", {}, agent=agent, backend=backend, attempts=2)
    assert len(backend.instructions) == 2
    assert isinstance(caught.value.__cause__, ValidationExhausted)
    assert "return 42" in caught.value.__cause__.output
    assert caught.value.__cause__.failure.measurement.passed is False


def test_source_execution_failure_not_retried():
    """A real failed agent command propagates without a new generation."""
    backend = RecordingLocalBackend()
    with pytest.raises(RuntimeError, match="Agent run failed"):
        agents.generate_source(
            "goal",
            {},
            agent=AgentSpec(kind="oracle", solution="exit 7"),
            backend=backend,
        )
    assert len(backend.instructions) == 1


@pytest.mark.parametrize("attempts", [0, -1, True, 1.5])
def test_source_bad_attempt_budget(attempts):
    """Budget errors are rejected before any agent work."""
    backend = RecordingLocalBackend()
    with pytest.raises(ValueError):
        agents.generate_source(
            "goal", {}, agent=AgentSpec(kind="nop"), backend=backend, attempts=attempts
        )
    assert backend.instructions == []


def test_source_empty_output_is_valid():
    """A valid empty list is accepted on the first attempt."""
    backend = RecordingLocalBackend()
    code = agents.generate_source(
        "empty",
        {},
        agent=AgentSpec(
            kind="oracle", solution="echo 'def run(item): return []' > program.py"
        ),
        backend=backend,
    )
    assert backend.execute_program(code, {}, 10) == []
    assert len(backend.instructions) == 1
