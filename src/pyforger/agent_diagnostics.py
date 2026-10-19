"""JSON-safe SDK event retention with credential redaction."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import Any

from pyforger.agent_backend import AgentSpec

_SENSITIVE = re.compile(
    r"api.?key|token|secret|password|authorization|credential", re.IGNORECASE
)
_ASSIGNMENT = re.compile(
    r"(?i)((?:[\w-]*(?:api[_-]?key|token|secret|password|authorization|credential)"
    r"[\w-]*)"
    r"[\"']?\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;}]+)"
)


def trajectory(events: Iterable[Any], spec: AgentSpec) -> dict[str, Any]:
    """Retain native event dictionaries, never SDK configuration or credentials."""
    secrets = {
        value for key, value in os.environ.items() if _SENSITIVE.search(key) and value
    }
    if spec.model is not None and spec.model.config.api_key:
        secrets.add(spec.model.config.api_key)
    if spec.api_key_env and os.environ.get(spec.api_key_env):
        secrets.add(os.environ[spec.api_key_env])

    def redact(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: "[REDACTED]" if _SENSITIVE.search(key) else redact(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, str):
            for secret in sorted(secrets, key=len, reverse=True):
                value = value.replace(secret, "[REDACTED]")
            value = re.sub(r"(?i)(Bearer|Basic)\s+[^\s\"']+", r"\1 [REDACTED]", value)
            return _ASSIGNMENT.sub(lambda m: m[1] + "[REDACTED]", value)
        return value

    return {
        "steps": [
            redact(event if isinstance(event, dict) else event.model_dump(mode="json"))
            for event in events
        ]
    }
