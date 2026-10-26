"""Explicit OpenAI-compatible endpoint model construction."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit

from pyforger.acquire.model import Model


def compatible_model(
    model: str,
    base_url: str | None,
    *,
    api_key_env: str = "LLM_API_KEY",
    extra_body: dict[str, Any] | None = None,
    **settings: Any,
) -> Model:
    """Construct a common Model, scoping known proxy quirks to its exact host."""
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("A model endpoint is required")
    endpoint = urlsplit(base_url)
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
        raise ValueError("A model endpoint is required")
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError("Credentials cannot be embedded in endpoint URLs")
    model_id = (
        "openai/" + model
        if "/" not in model or model.lower().startswith(("qwen/", "meta/"))
        else model
    )
    compatibility: dict[str, Any] = {}
    if endpoint.hostname == "llm-proxy.app.all-hands.dev":
        compatibility = {
            "disable_prompt_cache_key": True,
            "api_mode": "responses"
            if model.removeprefix("openai/")
            in {"gpt-6-luna", "deepseek-v4.1-flash", "gemini-3.5-flash"}
            else "auto",
        }
    return Model(
        model_id=model_id,
        api_key=os.environ.get(api_key_env),
        api_base=base_url,
        extra_body=extra_body or {},
        **{**compatibility, **settings},
    )
