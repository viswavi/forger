"""Input-specific OpenHands programs, cached as source and replayed on every call.

RestrictedPython is an accidental-damage guardrail, not a hostile-code security
boundary. Agents receive read-only URL and Git inspection tools, while generated
Python runs without terminal, filesystem, network, or unrestricted Python access.
"""

from __future__ import annotations

import json
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
    from openhands.sdk import LLM

    from pyforger._sdk_llm import ConfiguredLLM
    from pyforger.sdk_config import llm_options

    llm_type = ConfiguredLLM if model._kwargs.get("disable_prompt_cache_key") else LLM
    return llm_type(**llm_options(model))
