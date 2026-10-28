"""Own native asynchronous work through cancellation and mandatory cleanup."""

from __future__ import annotations

import asyncio
import contextvars
import time
from contextlib import contextmanager
from typing import Any, Callable, Coroutine, Iterator, TypeVar, cast
from weakref import WeakKeyDictionary

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


async def settle(tasks: set[asyncio.Task[Any]], end: float) -> None:
    # Native cancellation recovers in-flight create handles before stopping VMs.
    while pending := {task for task in tasks if not task.done()}:
        left = end - time.monotonic()
        if left <= 0:
            for task in pending:
                task.cancel()
            _, pending = await asyncio.wait(pending, timeout=1)
            if pending:
                raise RuntimeError("Harbor mandatory cleanup did not settle")
            raise RuntimeError("Harbor mandatory cleanup exceeded grace")
        await asyncio.wait(pending, timeout=left, return_when=asyncio.FIRST_COMPLETED)
    for task in tasks:
        if not task.cancelled():
            task.exception()


async def run_owned(
    operation: Callable[[], Coroutine[Any, Any, _T]], deadline: float | None
) -> _T:
    remaining(deadline)
    with owned_tasks() as tasks:
        root = asyncio.create_task(operation())
        try:
            done, _ = await asyncio.wait({root}, timeout=remaining(deadline))
            if not done:
                raise TimeoutError("Harbor end-to-end deadline exhausted")
            result = root.result()
            remaining(deadline)
        except BaseException as error:
            root.cancel()
            cleanup = asyncio.create_task(
                settle(tasks, time.monotonic() + CLEANUP_GRACE_S)
            )
            tasks.discard(cleanup)
            # Retain ownership even if the caller repeats cancellation during teardown.
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
            try:
                cleanup.result()
            except Exception as cleanup_error:
                setattr(
                    cleanup_error,
                    "diagnostics",
                    {
                        "deadline_exhausted": True,
                        "cleanup": "failed",
                        "cleanup_grace_s": CLEANUP_GRACE_S,
                    },
                )
                raise
            setattr(
                error,
                "diagnostics",
                {
                    "deadline_exhausted": isinstance(error, TimeoutError),
                    "cleanup": "settled",
                    "cleanup_grace_s": CLEANUP_GRACE_S,
                },
            )
            raise
        await settle(tasks, time.monotonic() + CLEANUP_GRACE_S)
        remaining(deadline)
        return result
