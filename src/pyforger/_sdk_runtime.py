"""Shared opt-in history configuration for SDK runners."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from litellm.exceptions import BadRequestError
from openhands.sdk import LLM, ToolDefinition
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.context.view import View
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.llm import LLMResponse, Message
from openhands.sdk.llm.exceptions import LLMBadRequestError, LLMContextWindowExceedError
from pydantic import Field

if TYPE_CHECKING:
    from pyforger.agent_backend import AgentSpec


class PromptBudgetLLM(LLM):
    """Route full-request pressure into the SDK's task-preserving recovery loop."""

    prompt_token_limit: int | None = Field(default=None, gt=0)

    def _check_prompt_budget(
        self,
        messages: list[Message],
        tools: Sequence[ToolDefinition[Any, Any]] | None,
        add_security_risk_prediction: bool,
    ) -> None:
        if not tools or self.prompt_token_limit is None:
            return
        count = self.get_token_count(
            messages,
            tools=tools,
            add_security_risk_prediction=add_security_risk_prediction,
        )
        if count <= 0:
            raise ValueError("Cannot measure serialized prompt token budget")
        if count > self.prompt_token_limit:
            raise LLMContextWindowExceedError(
                f"Serialized prompt has {count} tokens, exceeding {self.prompt_token_limit}"
            )

    @staticmethod
    def _raise_context_overflow(exc: Exception) -> None:
        # This gateway wording is not classified by SDK 1.58/LiteLLM.
        if isinstance(exc, (BadRequestError, LLMBadRequestError)) and (
            "requested token count exceeds the model's maximum context length"
            in str(exc).lower()
        ):
            raise LLMContextWindowExceedError(str(exc)) from exc

    def generate(
        self,
        messages: list[Message],
        tools: Sequence[ToolDefinition[Any, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self._check_prompt_budget(
            messages, tools, kwargs.get("add_security_risk_prediction", False)
        )
        try:
            return super().generate(messages=messages, tools=tools, **kwargs)
        except Exception as exc:
            self._raise_context_overflow(exc)
            raise

    async def agenerate(
        self,
        messages: list[Message],
        tools: Sequence[ToolDefinition[Any, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self._check_prompt_budget(
            messages, tools, kwargs.get("add_security_risk_prediction", False)
        )
        try:
            return await super().agenerate(messages=messages, tools=tools, **kwargs)
        except Exception as exc:
            self._raise_context_overflow(exc)
            raise


class TaskPreservingCondenser(LLMSummarizingCondenser):
    """Keep the system and original task even during SDK hard-reset recovery."""

    # The SDK observability decorator erases this method's static signature.
    def hard_context_reset(  # type: ignore[override]
        self, view: View, agent_llm: LLM | None = None
    ) -> Condensation | None:
        """Summarize only events after the protected prefix."""
        if len(view.events) <= self.keep_first:
            raise ValueError("Cannot condense the protected system/task prefix")
        events = view.events[self.keep_first :]
        limit = None
        for _ in range(self.hard_context_reset_max_retries):
            try:
                return self._generate_condensation(
                    events, self.keep_first, max_event_str_length=limit
                )
            except Exception:
                limit = int(
                    (limit if limit is not None else max(len(str(e)) for e in events))
                    * self.hard_context_reset_context_scaling
                )
        return None

    async def ahard_context_reset(
        self, view: View, agent_llm: LLM | None = None
    ) -> Condensation | None:
        """Preserve the same prefix in asynchronous conversations."""
        if len(view.events) <= self.keep_first:
            raise ValueError("Cannot condense the protected system/task prefix")
        events = view.events[self.keep_first :]
        limit = None
        for _ in range(self.hard_context_reset_max_retries):
            try:
                return await self._agenerate_condensation(
                    events, self.keep_first, max_event_str_length=limit
                )
            except Exception:
                limit = int(
                    (limit if limit is not None else max(len(str(e)) for e in events))
                    * self.hard_context_reset_context_scaling
                )
        return None


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
