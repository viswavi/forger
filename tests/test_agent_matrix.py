"""Every agent function runs on every backend, without LLM calls.

Deterministic oracle agents make the function-by-backend matrix executable:
the same AgentSpec and the same function calls must work unchanged on the
local backend and on the Harbor backend (Docker).
"""

from __future__ import annotations

import json
import os

import pytest

from pyforger import agents
from pyforger.agent_backend import AgentSpec, WorkspaceTask
from pyforger.agent_tasks import TaskSpec, run_command, run_task
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


def test_run_task_with_verifier(backend):
    """Verify run task with verifier."""
    task = TaskSpec(
        instruction="write answer.json",
        files={"seed.txt": "1"},
        outputs=("answer.json",),
        verifier="test -f answer.json && echo 1 > reward.txt || echo 0 > reward.txt",
    )
    outcome = run_task(
        task,
        agent=AgentSpec(kind="oracle", solution="cp seed.txt answer.json"),
        backend=backend,
    )
    assert outcome["status"] == "completed"
    assert outcome["passed"] is True
    assert outcome["files"]["answer.json"] == b"1"


def test_review_as_ordinary_task(backend):
    """Review uses caller-defined task instructions and report formats."""
    outcome = run_task(
        TaskSpec(
            instruction="Inspect answer.json against rubric.md and write review.json",
            files={"answer.json": "1", "rubric.md": "check: answer_present"},
            outputs=("review.json",),
        ),
        agent=AgentSpec(
            kind="oracle",
            solution=(
                "cat > review.json <<'EOF'\n"
                '{"checks": {"answer_present": {"outcome": "pass"}}}\n'
                "EOF\n"
            ),
        ),
        backend=backend,
    )
    report = json.loads(outcome["files"]["review.json"])
    assert report["checks"]["answer_present"]["outcome"] == "pass"


@pytest.mark.parametrize(
    "command",
    ["printf ok > answer.txt", "sh solve.sh", "python3 solve.py"],
)
def test_run_command_with_verifier(backend, command):
    """Commands of different kinds use the task verifier and declared outputs."""
    task = TaskSpec(
        instruction="produce an answer",
        files={
            "solve.sh": "printf ok > answer.txt\n",
            "solve.py": "open('answer.txt', 'w').write('ok')\n",
        },
        outputs=("answer.txt",),
        verifier='test "$(cat answer.txt)" = ok && printf 1 > reward.txt',
    )
    outcome = run_command(task, command, backend=backend)
    assert outcome["status"] == "completed"
    assert outcome["passed"] is True
    assert outcome["files"]["answer.txt"] == b"ok"


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
