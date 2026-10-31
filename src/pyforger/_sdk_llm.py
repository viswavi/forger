"""SDK prompt-cache compatibility shared with the native Harbor runner."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from openhands.sdk import LLM, ToolDefinition
from openhands.sdk.llm import LLMCallContext, LLMResponse, Message, TokenCallbackType


class ConfiguredLLM(LLM):
    """SDK LLM suppressing conversation-specific prompt cache keys."""

    def generate(
        self,
        messages: list[Message],
        tools: Sequence[ToolDefinition[Any, Any]] | None = None,
        include: list[str] | None = None,
        store: bool | None = None,
        add_security_risk_prediction: bool = False,
        on_token: TokenCallbackType | None = None,
        call_context: LLMCallContext | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Clear the cache key while preserving the rest of the call context."""
        call_context = replace(call_context or LLMCallContext(), prompt_cache_key=None)
        return super().generate(
            messages=messages,
            tools=tools,
            include=include,
            store=store,
            add_security_risk_prediction=add_security_risk_prediction,
            on_token=on_token,
            call_context=call_context,
            **kwargs,
        )
