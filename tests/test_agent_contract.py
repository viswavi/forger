"""Static and runtime contract for agent functions over any backend."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from pyforger.agent_backend import AgentBackend, AgentSpec
from pyforger.local_backend import LocalBackend


def test_agent_spec_validation():
    """Verify agent spec validation."""
    with pytest.raises(ValueError, match="require a Model"):
        AgentSpec()
    with pytest.raises(ValueError, match="oracle agents require"):
        AgentSpec(kind="oracle")
    with pytest.raises(ValueError, match="agent kind"):
        AgentSpec(kind="other", solution="true")
    assert AgentSpec(kind="nop").cache_key()


def test_backends_satisfy_protocol_without_models():
    """Verify backends satisfy protocol without models."""
    for backend in (LocalBackend(),):
        assert isinstance(backend, AgentBackend)
        assert backend.cache_key()
        assert not hasattr(backend, "model")


def test_functions_accept_any_backend_statically(tmp_path):
    """Mypy proves agent functions compose with every backend type."""
    source = tmp_path / "agent_contract.py"
    source.write_text(
        "from pyforger import agents\n"
        "from pyforger.acquire.model import Model\n"
        "from pyforger.agent_backend import AgentBackend, AgentSpec\n"
        "from pyforger.local_backend import LocalBackend\n"
        "agent = AgentSpec(Model('openai/test'))\n"
        "backends: list[AgentBackend] = [LocalBackend()]\n"
        "for backend in backends:\n"
        "    agents.generate_source('g', {}, agent=agent, backend=backend)\n"
        "    agents.generate_files('i', {}, ('o',), agent=agent, backend=backend)\n"
        "    agents.execute_source('c', {}, backend=backend)\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--ignore-missing-imports",
            "--cache-dir",
            str(tmp_path / "mypy-cache"),
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).parents[1],
    )
    assert result.returncode == 0, result.stdout + result.stderr
