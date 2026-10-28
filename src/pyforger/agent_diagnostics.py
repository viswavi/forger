"""JSON-safe SDK event retention with credential redaction."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from pyforger.artifacts import FileInput

if TYPE_CHECKING:
    from pyforger.agent_backend import AgentSpec

_SENSITIVE = re.compile(
    r"api.?key|token|secret|password|authorization|credential", re.IGNORECASE
)
_ASSIGNMENT = re.compile(
    r"(?i)(?<![\w-])((?:[\w-]*(?:api[_-]?key|token|secret|password|authorization|credential)"
    r"[\w-]*)"
    r"[\"']?\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;}]+)"
)


def credential_values(specs: Iterable[AgentSpec]) -> set[str]:
    secrets = {
        value for key, value in os.environ.items() if _SENSITIVE.search(key) and value
    }
    for spec in specs:
        if spec.model is not None and spec.model.config.api_key:
            secrets.add(spec.model.config.api_key)
        if spec.api_key_env and os.environ.get(spec.api_key_env):
            secrets.add(os.environ[spec.api_key_env])
        secrets.update(getattr(spec, "secrets", ()))
    return secrets


class ArtifactSafetyError(ValueError):
    """Declared output contains a concrete credential and cannot be published."""


def validate_artifact_bytes(
    files: Mapping[str, str | bytes | FileInput], specs: Iterable[AgentSpec]
) -> None:
    """Check exact bytes only; assignment-like scientific source is not a leak."""
    secrets = {value.encode("utf-8") for value in credential_values(specs) if value}
    overlap = max(map(len, secrets), default=1) - 1
    for content in files.values():
        if isinstance(content, str):
            content = content.encode("utf-8")
        if not isinstance(content, (bytes, FileInput)):
            raise TypeError("Malformed output artifact")
        tail = b""
        chunks = content.chunks() if isinstance(content, FileInput) else (content,)
        for chunk in chunks:
            block = tail + chunk
            if any(secret in block for secret in secrets):
                raise ArtifactSafetyError(
                    "Credential detected in output artifact. Remove credential literals "
                    "from all declared outputs; do not echo credentials in feedback."
                )
            tail = block[-overlap:] if overlap else b""


def sanitizer(specs: Iterable[AgentSpec], *, secrets: Iterable[str] = ()):
    """Snapshot every currently known concrete credential at persistence ingress."""
    secrets = credential_values(specs) | set(secrets)

    def redact(value: Any) -> Any:
        if isinstance(value, dict):
            # Preserve private capabilities across metadata-only sanitation.
            try:
                from pyforger.evidence_journal import is_private_reference
            except ImportError:
                is_private_reference = lambda _: False
            if is_private_reference(value):
                return value
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

    return redact


def trajectory(events: Iterable[Any], spec: AgentSpec) -> dict[str, Any]:
    """Legacy in-memory reader; use the evidence journal for execution histories."""
    clean = sanitizer([spec])
    return {
        "steps": [
            clean(event if isinstance(event, dict) else event.model_dump(mode="json"))
            for event in events
        ]
    }
