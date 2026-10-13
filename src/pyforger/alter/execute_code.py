"""Run each item's code snippet through a Sandbox; record the result."""
from __future__ import annotations

from pyforger.executors import PythonExecutor
from pyforger.op import op
from pyforger.sandbox import Sandbox
from pyforger.utils import get_field


def _execute_rows(
    items: list[dict],
    *,
    code_field: str,
    arg_field: str,
    output_field: str,
    error_field: str | None,
    sb: PythonExecutor,
    timeout_s: float | None,
) -> list[dict]:
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


@op(name="alter.execute_code")
def _cached_execute_code(
    items: list[dict],
    *,
    code_field: str,
    arg_field: str,
    output_field: str = "result",
    error_field: str | None = "error",
    sandbox: Sandbox | None = None,
    timeout_s: float | None = None,
) -> list[dict]:
    return _execute_rows(
        items,
        code_field=code_field,
        arg_field=arg_field,
        output_field=output_field,
        error_field=error_field,
        sb=sandbox if sandbox is not None else Sandbox(),
        timeout_s=timeout_s,
    )


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

    ``sandbox`` is a legacy alias, mutually exclusive with ``executor``. The
    Sandbox path keeps the historical ``@op`` caching; an explicit ``executor``
    is never implicitly cached, since executor state and stochastic source may
    change between calls. Timeout policy is executor-owned.
    """
    if sandbox is not None and executor is not None:
        raise ValueError("sandbox and executor are mutually exclusive")
    if executor is None:
        return _cached_execute_code(
            items,
            code_field=code_field,
            arg_field=arg_field,
            output_field=output_field,
            error_field=error_field,
            sandbox=sandbox,
            timeout_s=timeout_s,
        )
    return _execute_rows(
        items,
        code_field=code_field,
        arg_field=arg_field,
        output_field=output_field,
        error_field=error_field,
        sb=executor,
        timeout_s=timeout_s,
    )
