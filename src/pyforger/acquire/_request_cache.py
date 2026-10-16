"""Per-completion disk cache behind ``Model.generate_batch``.

``@op`` caches a whole call, so one failed request (or a killed run) used to
re-run every request of the op. This cache keeps each finished completion, so a
rerun only pays for the completions that never arrived.

Layout: ``$FORGER_CACHE/requests/<key[:2]>/<key>.<slot>.json``. ``key`` hashes
the op call the request comes from (op, body, every argument but the rows),
the row the prompt was built from, the messages, every output-changing request
param, and an optional salt; ``slot`` numbers the completions of that request
in the op call (n=4 uses slots 0..3; asking again takes 4..7).

One small JSON file per slot: writes are a temp file plus ``os.replace``, so
threads and processes need no lock, and a slot written twice at once keeps the
last write. The 256 shard directories keep each directory small (a million
completions is ~4,000 files per directory). JSON, unlike the op cache's pickle,
cannot execute code when read.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pyforger.op import _atomic_write, _cache_dir, _hash
from pyforger.utils import get_formatted_logger

logger = get_formatted_logger(__name__)

# Transport knobs that cannot change a completion: the API key, timeouts,
# retries, and logging metadata. Everything else Model passes to
# litellm.completion is part of the key: temperature, max_tokens, stop, seed,
# reasoning_effort, response_format, extra_body, and also api_base, base_url,
# api_version, and custom_llm_provider (two servers can serve different
# weights under one model name).
_UNKEYED_PARAMS = frozenset(
    {
        "api_key",
        "timeout",
        "request_timeout",
        "num_retries",
        "max_retries",
        "metadata",
    }
)

# Requests whose callers may read more than text from `raw` (tool calls,
# logprobs) are not cached: a replay has only the text.
_UNCACHED_IF_SET = ("tools", "functions", "tool_choice", "logprobs", "top_logprobs")

_KEY_VERSION = "forger-request-v2"


def enabled(model_flag: bool) -> bool:
    """True unless the Model or ``FORGER_REQUEST_CACHE=0`` turns the cache off."""
    flag = os.environ.get("FORGER_REQUEST_CACHE", "").strip().lower()
    return model_flag and flag not in {"0", "false", "no"}


def cacheable(call_kwargs: dict[str, Any]) -> bool:
    """False for requests that ask for tool calls or logprobs."""
    return not any(call_kwargs.get(name) for name in _UNCACHED_IF_SET)


def keyed_params(call_kwargs: dict[str, Any]) -> dict[str, Any]:
    """The request params that can change a completion (model id included)."""
    return {k: v for k, v in call_kwargs.items() if k not in _UNKEYED_PARAMS}


def request_key(
    scope: str,
    row: str,
    messages: list[dict[str, Any]],
    call_kwargs: dict[str, Any],
    salt: str | None,
) -> str:
    """Hex key for one prompt's completions.

    `scope` identifies the op call (op, body, non-row arguments), `row` the row
    the prompt was built from; `call_kwargs` includes the model id.
    """
    key, _ = _hash(
        [_KEY_VERSION, scope, row, messages, keyed_params(call_kwargs), salt]
    )
    return key


def _slot_path(key: str, slot: int) -> Path:
    return _cache_dir() / "requests" / key[:2] / f"{key}.{slot}.json"


def read_slot(key: str, slot: int) -> dict[str, Any] | None:
    """The cached completion for (key, slot), or None when absent or unusable."""
    path = _slot_path(key, slot)
    try:
        with path.open("rb") as f:
            # Same rule as the op cache: ignore files another user wrote.
            if hasattr(os, "getuid") and os.fstat(f.fileno()).st_uid != os.getuid():
                logger.debug("request cache file %s is owned by another user", path)
                return None
            entry = json.loads(f.read())
    except FileNotFoundError:
        return None
    except Exception as exc:  # unreadable entry: request the completion again
        logger.debug("request cache read failed for %s: %s", path, exc)
        return None
    if not isinstance(entry, dict) or not isinstance(entry.get("text"), str):
        return None
    return entry


def write_slot(key: str, slot: int, entry: dict[str, Any]) -> None:
    """Store one completion atomically; readers never see a partial file."""
    path = _slot_path(key, slot)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(entry).encode())
