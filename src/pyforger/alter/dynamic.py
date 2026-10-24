"""Cached dynamic alter operations over any agent and backend."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pyforger import agents
from pyforger.agent_backend import AgentBackend, AgentSpec
from pyforger.agents import _default
from pyforger.local_agent import _record_key
from pyforger.op import op

_SOURCE_VERSION = "agent-backend-v1"


@op(name="alter.dynamic_step_source")
def dynamic_step_source(
    item: dict[str, Any],
    goal: str,
    agent: AgentSpec,
    backend: AgentBackend | None = None,
    *,
    timeout_s: float = 10.0,
    source_version: str = _SOURCE_VERSION,
) -> str:
    """Cache generated source by agent and backend functional identity."""
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("goal must be nonempty")
    _record_key(item)
    if not isinstance(source_version, str) or not source_version:
        raise ValueError("source_version must be nonempty")
    return agents.generate_source(
        goal, item, agent=agent, backend=_default(backend), timeout_s=timeout_s
    )


@op(cache=False)
def dynamic_step(
    inputs: list[dict[str, Any]],
    goal: str,
    agent: AgentSpec,
    backend: AgentBackend | None = None,
    *,
    timeout_s: float = 10.0,
    max_parallel: int = 10,
    source_version: str = _SOURCE_VERSION,
) -> list[list[str]]:
    """Build source per unique input with any agent on any backend."""
    if not isinstance(inputs, list):
        raise TypeError("inputs must be a list of dicts")
    keys = [_record_key(item) for item in inputs]
    if (
        isinstance(max_parallel, bool)
        or not isinstance(max_parallel, int)
        or max_parallel < 1
    ):
        raise ValueError("max_parallel must be a positive integer")
    if not inputs:
        return []

    def run(item: dict[str, Any]) -> list[str]:
        code = dynamic_step_source(
            item,
            goal,
            agent,
            backend,
            timeout_s=timeout_s,
            source_version=source_version,
        )
        return _default(backend).execute_program(code, item, timeout_s)

    unique = dict(zip(keys, inputs))
    with ThreadPoolExecutor(max_workers=min(max_parallel, len(unique))) as pool:
        outputs = dict(zip(unique, pool.map(run, unique.values())))
    return [outputs[key].copy() for key in keys]
