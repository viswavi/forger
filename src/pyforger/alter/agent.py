"""Uncached row operations for separately executing and grading agents."""
from __future__ import annotations

import hashlib
from pathlib import Path

from pyforger.trials import RolloutResult, TrialRequest
from pyforger.trials import evaluate_agent as evaluate_one
from pyforger.trials import execute_agent as execute_one
from pyforger.trials import replay_agent as replay_one
from pyforger.utils import get_field


def _requests(items, request_field):
    requests = []
    for item in items:
        value = get_field(item, request_field, "alter.agent")
        requests.append(
            value if isinstance(value, TrialRequest) else TrialRequest.from_dict(value)
        )
    identities = [request.trial_id for request in requests]
    if len(set(identities)) != len(identities):
        raise ValueError("Each row requires a unique explicit trial_id")
    return requests


def _output(root, request):
    return Path(root) / hashlib.sha256(request.trial_id.encode()).hexdigest()


def execute_agent(
    items: list[dict],
    *,
    runner,
    output: Path,
    request_field: str = "request",
    output_field: str = "rollout",
    environment_factory=None,
) -> list[dict]:
    """Execute explicit row requests once; never implicitly cache samples."""
    return [
        {
            **item,
            output_field: execute_one(
                request,
                runner,
                _output(output, request),
                environment_factory=environment_factory,
            ).to_dict(),
        }
        for item, request in zip(items, _requests(items, request_field))
    ]


def evaluate_agent(
    items: list[dict],
    *,
    evaluator,
    output: Path,
    request_field: str = "request",
    rollout_field: str = "rollout",
    output_field: str = "evaluation",
) -> list[dict]:
    """Grade retained row rollouts independently of agent execution."""
    return [
        {
            **item,
            output_field: evaluate_one(
                request,
                RolloutResult.from_dict(get_field(item, rollout_field, "alter.agent")),
                evaluator,
                _output(output, request),
            ).to_dict(),
        }
        for item, request in zip(items, _requests(items, request_field))
    ]


def replay_agent(
    items: list[dict],
    *,
    runner,
    output: Path,
    environment_factory=None,
    request_field: str = "request",
    rollout_field: str = "rollout",
    output_field: str = "replay",
) -> list[dict]:
    """Replay retained row programs in newly opened environments."""
    return [
        {
            **item,
            output_field: replay_one(
                request,
                RolloutResult.from_dict(get_field(item, rollout_field, "alter.agent")),
                runner,
                _output(output, request),
                environment_factory=environment_factory,
            ).to_dict(),
        }
        for item, request in zip(items, _requests(items, request_field))
    ]
