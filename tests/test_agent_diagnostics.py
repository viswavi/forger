"""Retained native SDK messages redact tool arguments and credentials."""

import json

from openhands.sdk import Message, TextContent
from openhands.sdk.event import MessageEvent

from pyforger.acquire.model import Model
from pyforger.agent_backend import AgentSpec, reward_passed
from pyforger.agent_diagnostics import trajectory


def test_native_event_redaction(monkeypatch):
    """Native event redaction."""
    monkeypatch.setenv("SERVICE_TOKEN", "environment-credential")
    event = MessageEvent(
        source="agent",
        llm_message=Message(
            role="assistant",
            content=[
                TextContent(
                    text=(
                        'command: curl -H "Authorization: Bearer private-value"; '
                        'API_KEY="inline-private-value"; '
                        "environment-credential model-credential"
                    )
                )
            ],
        ),
    )
    result = trajectory(
        [event], AgentSpec(Model("openai/test", api_key="model-credential"))
    )
    serialized = json.dumps(result)
    for value in [
        "private-value",
        "inline-private-value",
        "environment-credential",
        "model-credential",
    ]:
        assert value not in serialized
    assert result["steps"][0]["kind"] == "MessageEvent"
    assert "[REDACTED]" in serialized


def test_numeric_rewards():
    """Numeric rewards."""
    assert all(reward_passed(value) for value in [b"1", b"1.0\n", b"1.00"])
    assert not any(
        reward_passed(value) for value in [b"", b"0", b"nan", b"inf", b"true", b"1.1"]
    )
