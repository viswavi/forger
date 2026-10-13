"""Run each item's code snippet through a Sandbox; record the result."""
from __future__ import annotations

from pyforger.executors import PythonExecutor
from pyforger.op import op
from pyforger.sandbox import Sandbox
from pyforger.utils import get_field


@op(name="alter.execute_code", cache=False)
def execute_code(
    items: list[dict],
    *,
    code_field: str,
    arg_field: str,
    output_field: str = "result",
    error_field: str | None = "error",
    sandbox: Sandbox | None = None,
    executor: PythonExecutor | None = None,
    timeout_s: float | None = None,
) -> list[dict]:
    """Execute each snippet with a swappable executor and record row errors.

    ``sandbox`` is a legacy alias, mutually exclusive with ``executor``. Defaults
    to Sandbox. Operation results are never implicitly cached; executor state and
    stochastic source may change between calls. Timeout policy is executor-owned.
    """
    if sandbox is not None and executor is not None:
        raise ValueError("sandbox and executor are mutually exclusive")
    sb = (
        executor
        if executor is not None
        else sandbox
        if sandbox is not None
        else Sandbox()
    )
    out: list[dict] = []
    for item in items:
        code = get_field(item, code_field, "alter.execute_code")
        arg = get_field(item, arg_field, "alter.execute_code")
        new_item = dict(item)
        try:
            new_item[output_field] = sb.execute(code, arg, timeout_s=timeout_s)
            if error_field is not None:
                new_item[error_field] = None
        except (KeyboardInterrupt, GeneratorExit):
            raise
        except BaseException as exc:
            new_item[output_field] = None
            if error_field is not None:
                new_item[error_field] = f"{type(exc).__name__}: {exc}"
        out.append(new_item)
    return out
