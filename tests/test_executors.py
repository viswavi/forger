"""Test strict JSON validation and real shared-environment Python execution."""
from __future__ import annotations

import os
import subprocess

import pytest

from pyforger.alter import execute_code
from pyforger.execution import DockerEnvironmentFactory, EnvironmentSpec
from pyforger.executors import (
    CodeExecutionError,
    ContainerPythonExecutor,
    DockerPythonExecutor,
    PythonExecutor,
)
from pyforger.sandbox import Sandbox, SandboxTimeout


@pytest.fixture
def container_executor():
    """Use the installed image only when Docker integration is explicitly enabled."""
    if os.environ.get("FORGER_DOCKER_TESTS") != "1":
        pytest.skip("requires FORGER_DOCKER_TESTS=1")
    image = os.environ.get("FORGER_DOCKER_IMAGE", "python:3.13-slim")
    try:
        checked = subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("Docker unavailable")
    if checked.returncode:
        pytest.skip("trusted image is not installed")
    return ContainerPythonExecutor(DockerEnvironmentFactory(), EnvironmentSpec(image))


@pytest.mark.parametrize(
    "value", [(1,), {1: "x"}, {"x": float("nan")}, float("inf"), b"x", {1}, object()]
)
def test_arguments_rejected_before_open(value):
    """Unsupported argument types are rejected without starting any environment."""
    executor = DockerPythonExecutor(
        DockerEnvironmentFactory(docker="nonexistent-docker")
    )
    with pytest.raises(TypeError, match="strict JSON"):
        executor.execute("def f(x): return x", value)


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf")])
def test_timeout_validation(timeout):
    """Invalid budgets cannot reach the environment factory."""
    executor = DockerPythonExecutor(
        DockerEnvironmentFactory(docker="nonexistent-docker")
    )
    with pytest.raises(ValueError, match="timeout_s"):
        executor.execute("def f(): return 1", timeout_s=timeout)


def test_spec_and_method_validation():
    """Conflicting specs and private method names fail clearly."""
    spec = EnvironmentSpec("python:3.13-slim")
    with pytest.raises(ValueError, match="either image or spec"):
        ContainerPythonExecutor(DockerEnvironmentFactory(), "other", spec=spec)
    with pytest.raises(ValueError, match="method name"):
        DockerPythonExecutor().call_method("", "Solution", "__init__")


def test_same_operation_rows(container_executor):
    """Sandbox and generic container executors produce identical successful rows."""
    code = (
        "def helper(x): return x + 1\ndef solve(x): return "
        "{'answer': [helper(x), x * 2]}"
    )
    rows = [{"code": code, "value": value} for value in (0, 7)]
    sandbox: PythonExecutor = Sandbox()
    remote: PythonExecutor = container_executor
    kwargs = {"code_field": "code", "arg_field": "value"}
    assert execute_code(rows, executor=sandbox, **kwargs) == execute_code(
        rows, executor=remote, **kwargs
    )


def test_last_function_kwargs_and_prints(container_executor):
    """Function selection and keyword arguments match Sandbox, independent of stdout."""
    code = (
        "def first(x): return -1\ndef last(x, scale=1):\n "
        "print('not protocol')\n return x * scale"
    )
    assert container_executor.execute(code, 7, scale=6) == 42


@pytest.mark.parametrize("executor_kind", ["sandbox", "container"])
def test_program_and_method(container_executor, executor_kind):
    """Both executors support script stdin, clean exits and fresh class instances."""
    executor = Sandbox() if executor_kind == "sandbox" else container_executor
    assert executor.run_program("print(int(input()) * 2)", stdin="21\n") == "42\n"
    assert executor.run_program("import sys\nprint('ok')\nsys.exit(0)") == "ok\n"
    code = "class Solution:\n def solve(self, x, scale=1): return x * scale"
    assert executor.call_method(code, "Solution", "solve", 7, scale=6) == 42


def test_fresh_environment(container_executor):
    """Every call starts with fresh filesystem and Python module state."""
    code = (
        "from pathlib import Path\ndef solve():\n "
        "p=Path('state')\n exists=p.exists()\n p.write_text('state')\n return exists"
    )
    assert container_executor.execute(code) is False
    assert container_executor.execute(code) is False
    code = "state=[]\ndef solve():\n state.append(1)\n return len(state)"
    assert container_executor.execute(code) == 1
    assert container_executor.execute(code) == 1


@pytest.mark.parametrize(
    "source,remote_type",
    [
        ("def f(): raise ValueError('bad')", "ValueError"),
        ("def f(: pass", "SyntaxError"),
        ("value = 1", "ValueError"),
        ("def f(): return (1, 2)", "TypeError"),
        ("def f(): return {1: 'bad'}", "TypeError"),
        ("def f(): return float('nan')", "TypeError"),
        ("def f(): return object()", "TypeError"),
    ],
)
def test_remote_errors(container_executor, source, remote_type):
    """Remote syntax, runtime and serialization failures retain diagnostic types."""
    with pytest.raises(CodeExecutionError) as error:
        container_executor.execute(source)
    assert error.value.remote_type == remote_type
    assert remote_type in error.value.traceback


def test_timeout_cleanup_and_reuse(container_executor):
    """Timeout terminates the environment but leaves the executor reusable."""
    with pytest.raises(SandboxTimeout):
        container_executor.execute("def f():\n while True: pass", timeout_s=0.1)
    assert container_executor.execute("def f(): return 42") == 42


def test_malicious_links_rejected(container_executor):
    """Generated filesystem links cannot bypass host-side output validation."""
    with pytest.raises(CodeExecutionError, match="unsafe output archive"):
        container_executor.execute(
            "from pathlib import Path\ndef f():\n "
            "Path('link').symlink_to('/etc/passwd')\n return 42"
        )


def test_fake_stdout_does_not_counterfeit_result(container_executor):
    """Printing a plausible protocol response does not change the returned value."""
    assert (
        container_executor.execute(
            'def f():\n print(\'{"ok": true, "value": 999}\')\n return 42'
        )
        == 42
    )
