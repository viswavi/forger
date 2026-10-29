"""Own native asynchronous work through cancellation and mandatory cleanup."""

from __future__ import annotations

import asyncio
import contextvars
import time
from contextlib import contextmanager
from typing import Any, Callable, Coroutine, Iterator, TypeVar, cast
from weakref import WeakKeyDictionary

IMAGE_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "sail_image_deadline", default=None
)

CLEANUP_GRACE_S = 45.0
_OWNER: contextvars.ContextVar[set[asyncio.Task[Any]] | None] = contextvars.ContextVar(
    "forger_harbor_owner", default=None
)
_FACTORIES: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()
_T = TypeVar("_T")


def remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("Harbor end-to-end deadline exhausted")
    return value


@contextmanager
def owned_tasks() -> Iterator[set[asyncio.Task[Any]]]:
    loop = asyncio.get_running_loop()
    if loop not in _FACTORIES:
        previous = loop.get_task_factory()

        def factory(loop: Any, coro: Any, **kwargs: Any) -> asyncio.Task[Any]:
            task = cast(
                asyncio.Task[Any],
                (
                    previous(loop, coro, **kwargs)
                    if previous
                    else asyncio.Task(coro, loop=loop, **kwargs)
                ),
            )
            context = kwargs.get("context")
            owner = context.get(_OWNER) if context is not None else _OWNER.get()
            if owner is not None:
                owner.add(task)
            return task

        loop.set_task_factory(factory)
        _FACTORIES[loop] = [previous, 0]
    entry = _FACTORIES[loop]
    entry[1] += 1
    tasks: set[asyncio.Task[Any]] = set()
    token = _OWNER.set(tasks)
    try:
        yield tasks
    finally:
        _OWNER.reset(token)
        entry[1] -= 1
        if entry[1] == 0:
            loop.set_task_factory(entry[0])
            del _FACTORIES[loop]


def task_snapshot(tasks: set[asyncio.Task[Any]]) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        coro = task.get_coro()
        rows.append(
            {
                "coroutine": getattr(coro, "__qualname__", type(coro).__name__),
                "done": task.done(),
                "cancelled": task.cancelled(),
                "stack": [
                    {
                        "file": frame.f_code.co_filename,
                        "line": frame.f_lineno,
                        "function": frame.f_code.co_name,
                    }
                    for frame in task.get_stack(limit=6)
                ],
            }
        )
    return rows


class HarborCleanupError(RuntimeError):
    def __init__(self, message: str, **diagnostics: Any) -> None:
        super().__init__(message)
        self.diagnostics = {
            "cleanup": "failed",
            "cleanup_grace_s": CLEANUP_GRACE_S,
            **diagnostics,
        }


async def settle(
    tasks: set[asyncio.Task[Any]], end: float, root: asyncio.Task[Any]
) -> None:
    initial = task_snapshot({task for task in tasks if not task.done()})
    while pending := {task for task in tasks if not task.done()}:
        left = end - time.monotonic()
        if left <= 0:
            snapshot = task_snapshot(pending)
            for task in pending:
                task.cancel()
            _, pending = await asyncio.wait(pending, timeout=1)
            raise HarborCleanupError(
                "Harbor mandatory cleanup did not settle"
                if pending
                else "Harbor mandatory cleanup exceeded grace",
                pending_tasks=snapshot,
                settlement_start=initial,
            )
        await asyncio.wait(pending, timeout=left, return_when=asyncio.FIRST_COMPLETED)
    failures = []
    for task in tasks:
        if not task.cancelled():
            # Native phase tasks may fail and be deliberately handled by their parent.
            unobserved = getattr(task, "_log_traceback", True)
            error = task.exception()
            if error is not None and task is not root and unobserved:
                failures.append(
                    {
                        "coroutine": getattr(
                            task.get_coro(), "__qualname__", "unknown"
                        ),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
    if failures:
        raise HarborCleanupError(
            "Harbor owned cleanup task failed", failed_tasks=failures
        )


async def run_owned(
    operation: Callable[[], Coroutine[Any, Any, _T]], deadline: float | None
) -> _T:
    remaining(deadline)
    with owned_tasks() as tasks:
        root = asyncio.create_task(operation())
        error: BaseException | None = None
        result: Any = None
        try:
            done, _ = await asyncio.wait({root}, timeout=remaining(deadline))
            if not done:
                raise TimeoutError("Harbor end-to-end deadline exhausted")
            result = root.result()
            remaining(deadline)
        except BaseException as exc:
            error = exc
            root.cancel()
        cleanup = asyncio.create_task(
            settle(tasks, time.monotonic() + CLEANUP_GRACE_S, root)
        )
        tasks.discard(cleanup)
        # The same settlement path owns teardown on normal and exceptional exits.
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as exc:
                if error is None:
                    error = exc
                root.cancel()
            except Exception:
                break
        try:
            cleanup.result()
        except HarborCleanupError as cleanup_error:
            if error is not None:
                cleanup_error.diagnostics["primary_error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
            raise
        if error is not None:
            setattr(
                error,
                "diagnostics",
                {
                    "deadline_exhausted": isinstance(error, TimeoutError),
                    "cleanup": "settled",
                    "cleanup_grace_s": CLEANUP_GRACE_S,
                },
            )
            raise error
        remaining(deadline)
        return cast(_T, result)
