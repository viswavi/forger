"""Cached dynamic alter operations using the local agent backend."""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pyforger.acquire.model import Model
from pyforger.dynamic_backend import DynamicStepRunner
from pyforger.local_agent import LocalAgentRunner, _record_key
from pyforger.op import op

_SOURCE_VERSION = "resource-v4-record"


def _runner(
    model: Model | None,
    runner: DynamicStepRunner | None,
    max_iterations: int,
    execution_timeout_s: float,
) -> DynamicStepRunner:
    if runner is not None:
        if model is not None or max_iterations != 20 or execution_timeout_s != 10.0:
            raise ValueError(
                "An explicit runner owns its model and limits; configure them on the runner"
            )
        return runner
    if model is None:
        raise ValueError("Provide a model or an explicit runner")
    return LocalAgentRunner(
        model, max_iterations=max_iterations, execution_timeout_s=execution_timeout_s
    )


@op(name="alter.dynamic_step_source")
def dynamic_step_source(
    item: dict[str, Any],
    goal: str,
    model: Model | None = None,
    *,
    max_iterations: int = 20,
    execution_timeout_s: float = 10.0,
    source_version: str = _SOURCE_VERSION,
    runner: DynamicStepRunner | None = None,
) -> str:
    """Generate Python source once per input through Forger's standard @op cache.

    Call with ``_cache_only=True`` to inspect cached source without launching
    an agent when there is a cache miss.
    """
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("goal must be a nonempty string")
    _record_key(item)
    if not isinstance(source_version, str) or not source_version:
        raise ValueError("source_version must be a nonempty string")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    if not math.isfinite(execution_timeout_s) or execution_timeout_s <= 0:
        raise ValueError("execution_timeout_s must be positive and finite")
    return _runner(model, runner, max_iterations, execution_timeout_s).generate_source(
        goal, item
    )


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

    backend = _runner(model, runner, max_iterations, execution_timeout_s)

    def run(item: dict[str, Any]) -> list[str]:
        if runner is None:
            code = dynamic_step_source(
                item,
                goal,
                model,
                max_iterations=max_iterations,
                execution_timeout_s=execution_timeout_s,
                source_version=source_version,
            )
        else:
            code = dynamic_step_source(
                item,
                goal,
                source_version=source_version,
                runner=backend,
            )
        return backend.execute(code, item)

    unique = dict(zip(keys, inputs))
    with ThreadPoolExecutor(max_workers=min(max_parallel, len(unique))) as pool:
        outputs = dict(zip(unique, pool.map(run, unique.values())))
    return [outputs[key].copy() for key in keys]
