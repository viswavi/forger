"""Same-conversation repair with host validation and shared execution budgets."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from openhands.sdk import Agent
from pydantic import PrivateAttr

from pyforger.measurements import Measurement
from pyforger.session_budget import SessionBudget as SessionBudget
from pyforger.structured import (
    GenerationAttempt,
    GenerationInconclusive,
    ValidatedGeneration,
    ValidationExhausted,
)


class BudgetedAgent(Agent):
    """SDK agent whose native steps share a caller-owned budget."""

    _budget: SessionBudget = PrivateAttr()

    @property
    def prompt_dir(self) -> str:
        """Keep native prompts when subclassing the SDK agent outside its package."""
        return str(Path(inspect.getfile(Agent)).parent / "prompts")

    @property
    def budget(self) -> SessionBudget:
        """Return the budget shared across all conversation runs."""
        return self._budget

    def attach_budget(self, budget: SessionBudget) -> None:
        """Attach shared execution accounting without mutating frozen model fields."""
        self._budget = budget

    def step(  # type: ignore[override]
        self, conversation: Any, on_event: Any, on_token: Any = None
    ) -> None:
        """Charge synchronous SDK steps across all repair turns."""
        self.budget.consume_step()
        native_agent: Any = super()
        native_agent.step(conversation, on_event, on_token)
        self.budget.remaining_s()

    # SDK observe decorator currently erases the native method signature in mypy.
    async def astep(  # type: ignore[override]
        self, conversation: Any, on_event: Any, on_token: Any = None
    ) -> None:
        """Charge the native step before dispatching any model or tool work."""
        self.budget.consume_step()
        native_agent: Any = super()
        native_step = native_agent.astep
        await native_step(conversation, on_event, on_token)
        self.budget.remaining_s()


async def run_validated_session(
    session: Any,
    *,
    instruction: str,
    collect: Callable[[], dict[str, bytes] | Awaitable[dict[str, bytes]]],
    outputs: tuple[str, ...],
    validate: Callable[[dict[str, bytes]], Measurement],
    max_repairs: int,
    budget: SessionBudget,
) -> ValidatedGeneration[str, dict[str, bytes]]:
    """Validate snapshots on the host and feed rejection into the same session.

    The caller owns session/workspace lifecycle. Synchronous host callbacks
    must be bounded themselves; their elapsed time is charged to the deadline.
    """
    if type(max_repairs) is not int or max_repairs < 0:
        raise ValueError("max_repairs must be a nonnegative integer")
    attempts: list[GenerationAttempt[str, dict[str, bytes]]] = []
    for index in range(max_repairs + 1):
        budget.remaining_s()
        session.send_message(instruction)
        await asyncio.wait_for(session.arun(), timeout=budget.remaining_s())
        status = session.state.execution_status
        if getattr(status, "value", status) != "finished":
            raise RuntimeError(f"Agent execution did not finish: {status}")
        produced = collect()
        if inspect.isawaitable(produced):
            produced = await asyncio.wait_for(produced, timeout=budget.remaining_s())
        budget.remaining_s()
        produced = dict(produced)
        missing = [name for name in outputs if name not in produced]
        measurement = (
            Measurement(passed=False, reason=f"Missing declared outputs: {missing}")
            if missing
            else validate(produced)
        )
        budget.remaining_s()
        if not isinstance(measurement, Measurement):
            raise TypeError("validate must return a Measurement")
        attempts.append(GenerationAttempt(instruction, produced, measurement))
        history = tuple(attempts)
        if measurement.passed is True:
            return ValidatedGeneration(produced, history)
        if measurement.passed is None:
            raise GenerationInconclusive(history)
        if index == max_repairs:
            raise ValidationExhausted(history)
        instruction = (
            "The produced files failed validation. Repair them in this same "
            "workspace, preserving useful work. Validation feedback:\n"
            + measurement.model_dump_json()
        )
    raise AssertionError("unreachable")
