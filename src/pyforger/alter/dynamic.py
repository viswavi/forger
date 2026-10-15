"""Cached dynamic alter operations with interchangeable execution backends."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pyforger.acquire.model import Model
from pyforger.dynamic_backend import DynamicStepRunner
from pyforger.local_agent import LocalAgentRunner, _record_key
from pyforger.op import op

_SOURCE_VERSION = "resource-v4-record"


def _default_runner(
    model: Model | None,
    max_iterations: int,
    execution_timeout_s: float,
) -> LocalAgentRunner:
    if model is None:
        raise ValueError("Provide a model or an explicit runner")
    return LocalAgentRunner(
        model, max_iterations=max_iterations, execution_timeout_s=execution_timeout_s
    )


@op(name="alter.dynamic_step_source")
def dynamic_step_source(
    item: dict[str, Any],
    goal: str,
    runner: DynamicStepRunner,
    *,
    source_version: str = _SOURCE_VERSION,
) -> str:
    """Cache generated source using the backend's functional identity.

    The runner is resolved by the caller. ``cache_only`` inspects retained
    source without constructing or running another backend.
    """
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("goal must be nonempty")
    _record_key(item)
    if not isinstance(source_version, str) or not source_version:
        raise ValueError("source_version must be nonempty")
    return runner.generate_source(goal, item)


@op(cache=False)
def dynamic_step(
    inputs: list[dict[str, Any]],
    goal: str,
    model: Model | None = None,
    *,
    max_parallel: int = 10,
    max_iterations: int = 20,
    execution_timeout_s: float = 10.0,
    source_version: str = _SOURCE_VERSION,
    runner: DynamicStepRunner | None = None,
) -> list[list[str]]:
    """Build source per unique input and return one string list per input."""
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

    runner = runner or _default_runner(model, max_iterations, execution_timeout_s)

    def run(item: dict[str, Any]) -> list[str]:
        code = dynamic_step_source(item, goal, runner, source_version=source_version)
        return runner.execute(code, item)

    unique = dict(zip(keys, inputs))
    with ThreadPoolExecutor(max_workers=min(max_parallel, len(unique))) as pool:
        outputs = dict(zip(unique, pool.map(run, unique.values())))
    return [outputs[key].copy() for key in keys]
