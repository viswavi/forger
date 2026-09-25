"""Run each item's code snippet through a Sandbox; record the result."""
from __future__ import annotations

from pyforger.op import op
from pyforger.sandbox import Sandbox
from pyforger.utils import get_field


@op(name="alter.execute_code")
def execute_code(
    items: list[dict],
    *,
    code_field: str,
    arg_field: str,
    output_field: str = "result",
    error_field: str | None = "error",
    sandbox: Sandbox | None = None,
    timeout_s: float | None = None,
) -> list[dict]:
    """Compile each item's code and call it with ``arg_field``'s value.

    The Sandbox caches by source string, so a snippet reused across items
    pays compile cost once. Exceptions land in ``error_field`` (set to
    ``None`` to omit) rather than raising, so one bad row doesn't kill
    the batch.

    ``timeout_s`` is the per-item wall-clock cap: a snippet that runs
    longer raises ``SandboxTimeout``, caught here and recorded in
    ``error_field`` like any other exception. ``None`` falls back to
    the Sandbox's ``default_timeout_s``. SIGALRM-based, so Unix only
    and main thread only.
    """
    sb = sandbox if sandbox is not None else Sandbox()
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
