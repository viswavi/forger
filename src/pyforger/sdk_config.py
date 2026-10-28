"""Serialize SDK model options without constructing a tokenizer or LLM."""

import os
from pathlib import Path
from typing import Any


def llm_options(model: Any) -> dict[str, Any]:
    """Translate model settings without initializing an SDK model or tokenizer."""
    # Read the caller's .env without mutating process-wide environment in threads.
    from dotenv import dotenv_values

    dotenv = dotenv_values(Path.cwd() / ".env")
    config = model.config
    key = (
        config.api_key
        or os.environ.get("LITELLM_API_KEY")
        or dotenv.get("LITELLM_API_KEY")
    )

    return dict(
        model=config.model_id,
        custom_tokenizer=config.custom_tokenizer,
        temperature=config.temperature,
        base_url=config.api_base
        or os.environ.get("LITELLM_BASE_URL")
        or os.environ.get("LLM_BASE_URL")
        or dotenv.get("LITELLM_BASE_URL")
        or dotenv.get("LLM_BASE_URL"),
        api_key=key,
        max_output_tokens=config.max_tokens,
        # Model counts additional retries; the SDK counts total attempts.
        num_retries=max(1, config.num_retries + 1),
        timeout=config.timeout if config.timeout is not None else 300,
        reasoning_effort=config.reasoning_effort,
        litellm_extra_body=config.extra_body,
        api_mode=config.api_mode,
        capability_overrides=config.capability_overrides,
        **(
            {"max_context_tokens": config.max_context_tokens}
            if config.max_context_tokens is not None
            else {}
        ),
    )
