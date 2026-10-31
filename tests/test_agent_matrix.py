"""Every agent function runs on every backend, without LLM calls.

Deterministic oracle agents make the function-by-backend matrix executable:
the same AgentSpec and the same function calls must work unchanged on the
local backend and on the Harbor backend (Docker).
"""

from __future__ import annotations

import os

import pytest

from pyforger import agents
from pyforger.agent_backend import AgentSpec, WorkspaceTask
from pyforger.alter import dynamic_step
from pyforger.local_backend import LocalBackend

ORACLE_PROGRAM = AgentSpec(
    kind="oracle",
    solution=(
        "cat > program.py <<'EOF'\n"
        "def pipeline(item):\n"
        "    return [item['text'].upper()]\n"
        "EOF\n"
    ),
)


@pytest.fixture(params=["local", "harbor-docker"])
def backend(request):
    """Provide each backend implementation in turn."""
    if request.param == "local":
        return LocalBackend()
    if not os.environ.get("PYFORGER_HARBOR_DOCKER_TESTS"):
        pytest.skip("set PYFORGER_HARBOR_DOCKER_TESTS=1 for Docker backend tests")
    from pyforger.harbor_backend import HarborBackend

    return HarborBackend()


def test_backend_replays_program(backend):
    """Verify backend replays program."""
    code = "def pipeline(item):\n    return [item['text'].upper()]\n"
    assert backend.execute_program(code, {"text": "abc"}, 10.0) == ["ABC"]


def test_generate_source_with_oracle_agent(backend):
    """Verify generate source with oracle agent."""
    code = agents.generate_source(
        "uppercase the text", {"text": "abc"}, agent=ORACLE_PROGRAM, backend=backend
    )
    assert backend.execute_program(code, {"text": "xy"}, 10.0) == ["XY"]


def test_generate_files_with_oracle_agent(backend):
    """Verify generate files with oracle agent."""
    produced = agents.generate_files(
        "write the answer",
        {"input.txt": "hello"},
        ("answer.txt",),
        agent=AgentSpec(kind="oracle", solution="cp input.txt answer.txt"),
        backend=backend,
    )
    assert produced["answer.txt"] == b"hello"


def test_dynamic_step_with_oracle_agent(backend):
    """Verify dynamic step with oracle agent."""
    rows = dynamic_step(
        [{"text": "a"}, {"text": "b"}], "uppercase", ORACLE_PROGRAM, backend
    )
    assert rows == [["A"], ["B"]]


def test_run_agent_workspace_directly(backend):
    """Verify run agent workspace directly."""
    result = backend.run_agent(
        WorkspaceTask(
            instruction="copy",
            files={"a.txt": "x"},
            outputs=("b.txt",),
        ),
        AgentSpec(kind="oracle", solution="cp a.txt b.txt"),
    )
    assert result.status == "completed"
    assert result.files["b.txt"] == b"x"
