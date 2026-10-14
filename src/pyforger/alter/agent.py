"""Uncached row operations for executing fresh agent samples."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pyforger.trials import AgentRunner, EnvironmentFactory, TrialRequest
from pyforger.trials import execute_agent as execute_one
from pyforger.utils import get_field


def _requests(items: list[dict[str, Any]], request_field: str) -> list[TrialRequest]:
    requests = []
    for item in items:
        value = get_field(item, request_field, "alter.agent")
        requests.append(
            value
            if isinstance(value, TrialRequest)
            else TrialRequest.model_validate(value)
        )
    identities = [request.trial_id for request in requests]
    if len(set(identities)) != len(identities):
        raise ValueError("Each row requires a unique explicit trial_id")
    return requests


def _output(root: Path, request: TrialRequest) -> Path:
    return Path(root) / hashlib.sha256(request.trial_id.encode()).hexdigest()


def execute_agent(
    items: list[dict[str, Any]],
    *,
    runner: AgentRunner,
    output: Path,
    request_field: str = "request",
    output_field: str = "rollout",
    environment_factory: EnvironmentFactory | None = None,
) -> list[dict[str, Any]]:
    """Execute explicit row requests once; never implicitly cache samples."""
    return [
        {
            **item,
            output_field: execute_one(
                request,
                runner,
                _output(output, request),
                environment_factory=environment_factory,
            ).model_dump(mode="json"),
        }
        for item, request in zip(items, _requests(items, request_field))
    ]
