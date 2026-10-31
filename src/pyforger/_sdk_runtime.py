"""Shared opt-in history configuration for SDK runners."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from openhands.sdk import LLM
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.context.view import View
from openhands.sdk.event.condenser import Condensation

if TYPE_CHECKING:
    from pyforger.agent_backend import AgentSpec


class TaskPreservingCondenser(LLMSummarizingCondenser):
    """Keep the system and original task even during SDK hard-reset recovery."""

    # The SDK observability decorator erases this method's static signature.
    def hard_context_reset(  # type: ignore[override]
        self, view: View, agent_llm: LLM | None = None
    ) -> Condensation | None:
        """Summarize only events after the protected prefix."""
        if len(view.events) <= self.keep_first:
            raise ValueError("Cannot condense the protected system/task prefix")
        return self._generate_condensation(
            view.events[self.keep_first :], self.keep_first
        )

    async def ahard_context_reset(
        self, view: View, agent_llm: LLM | None = None
    ) -> Condensation | None:
        """Preserve the same prefix in asynchronous conversations."""
        if len(view.events) <= self.keep_first:
            raise ValueError("Cannot condense the protected system/task prefix")
        return await self._agenerate_condensation(
            view.events[self.keep_first :], self.keep_first
        )


def sdk_history_kwargs(spec: AgentSpec, llm: LLM) -> dict[str, Any]:
    """Use the agent model for SDK token estimates and summary generation.

    The SDK removes oldest atomic turns towards half the token threshold before
    adding the summary. Counts use its tokenizer, not gateway usage accounting;
    the protected prefix and generated summary can exceed that target.
    """
    runtime = spec.runtime
    if runtime is None or runtime.condenser_max_tokens is None:
        return {}
    if "condenser" in spec.kwargs:
        raise ValueError("Agent kwargs conflict with the explicit condenser settings")
    return {
        "condenser": TaskPreservingCondenser(
            llm=llm,
            max_tokens=runtime.condenser_max_tokens,
            max_size=sys.maxsize,
            keep_first=2,
        )
    }


def sdk_iteration_limit(max_iterations: int | None) -> int:
    """Adapt optional budgets to SDK 1.53's positive-integer-only API.

    The SDK validates ConversationState.max_iterations as an int > 0 and
    compares its run counter directly against max_iteration_per_run. It has no
    unlimited mode. Only this internal boundary uses sys.maxsize; Forger keeps
    None and enforces the shared finite wall deadline on every step/turn.
    """
    return sys.maxsize if max_iterations is None else max_iterations
