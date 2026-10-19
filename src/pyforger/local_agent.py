"""Input-specific OpenHands programs, cached as source and replayed on every call.

RestrictedPython is an accidental-damage guardrail, not a hostile-code security
boundary. Agents receive read-only URL and Git inspection tools, while generated
Python runs without terminal, filesystem, network, or unrestricted Python access.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyforger.acquire.model import Model
from pyforger.agent_backend import ProgramValidationError
from pyforger.alter._dynamic_resources import prepare_input

if TYPE_CHECKING:
    from openhands.sdk import LLM


def _record_key(item: dict[str, Any]) -> str:
    """Validate a JSON record and give equal records the same batch key."""
    if not isinstance(item, dict):
        raise TypeError("each input must be a dict")
    try:
        encoded = json.dumps(
            item,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        if json.loads(encoded) != item:
            raise ValueError("input records must have string keys")
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "input records must be JSON-compatible dicts with string keys"
        ) from exc
    return encoded


def _require_dynamic_platform() -> None:
    """Fail before agent work on platforms without sandbox wall-clock timers."""
    if not all(
        hasattr(signal, name) for name in ("SIGALRM", "ITIMER_REAL", "setitimer")
    ):
        raise NotImplementedError("dynamic_step requires Unix signal.setitimer/SIGALRM")


def _execute(
    code: str,
    item: dict[str, Any],
    timeout: float,
    *,
    prepared_input: dict[str, Any] | None = None,
) -> list[str]:
    _require_dynamic_platform()
    # Do not inherit credentials into the generated-code process.
    env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    with tempfile.TemporaryDirectory(prefix="forger-execute-") as workspace:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "alter" / "_dynamic_worker.py"),
            ],
            input=json.dumps(
                {
                    "code": code,
                    "timeout_s": timeout,
                    "item": (
                        prepared_input
                        if prepared_input is not None
                        else prepare_input(item)
                    ),
                }
            ),
            text=True,
            capture_output=True,
            timeout=timeout + 2,
            cwd=workspace,
            env=env,
        )
    if result.returncode:
        raise ProgramValidationError(result.stderr[-4000:])
    output: object = json.loads(result.stdout)
    if not isinstance(output, list):
        raise TypeError("The pipeline must return a list of strings")
    strings: list[str] = []
    for value in output:
        if not isinstance(value, str):
            raise TypeError("The pipeline must return a list of strings")
        strings.append(value)
    return strings


def agent_llm(model: Model) -> LLM:
    """Construct the SDK model shared by dynamic and artifact-aware agents."""
    # Read the caller's .env without mutating process-wide environment in threads.
    from dotenv import dotenv_values
    from openhands.sdk import LLM
    from pydantic import SecretStr

    from pyforger._sdk_llm import ConfiguredLLM

    dotenv = dotenv_values(Path.cwd() / ".env")
    config = model.config
    key = (
        config.api_key
        or os.environ.get("LITELLM_API_KEY")
        or dotenv.get("LITELLM_API_KEY")
    )

    llm_type = ConfiguredLLM if model._kwargs.get("disable_prompt_cache_key") else LLM
    return llm_type(
        model=config.model_id,
        temperature=config.temperature,
        base_url=config.api_base
        or os.environ.get("LITELLM_BASE_URL")
        or os.environ.get("LLM_BASE_URL")
        or dotenv.get("LITELLM_BASE_URL")
        or dotenv.get("LLM_BASE_URL"),
        api_key=SecretStr(key) if key else None,
        max_output_tokens=config.max_tokens,
        # Model counts additional retries; the SDK counts total attempts.
        num_retries=max(1, config.num_retries + 1),
        timeout=config.timeout if config.timeout is not None else 300,
        reasoning_effort=config.reasoning_effort or "high",
        litellm_extra_body=config.extra_body,
        api_mode=config.api_mode,
        capability_overrides=config.capability_overrides,
    )
