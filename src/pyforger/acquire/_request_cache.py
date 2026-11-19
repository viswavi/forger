"""Per-completion disk cache behind ``Model.generate_batch``.

``@op`` caches a whole call, so one failed request (or a killed run) used to
re-run every request of the op. This cache keeps each finished completion, so a
rerun only pays for the completions that never arrived.

Layout: ``$FORGER_CACHE/requests/<key[:2]>/<key>.<slot>.json``. ``key`` hashes
the op call the request comes from (op, body, every argument but the rows),
the row the prompt was built from, the messages, every output-changing request
param, and an optional salt; ``slot`` numbers the completions of that request
in the op call (n=4 uses slots 0..3; asking again takes 4..7).

A slot whose request hit a row error (a content-filter block, an over-long
prompt) stores that error instead (``row_error_entry``), so a rerun replays it.

One small JSON file per slot: writes are a temp file plus ``os.replace``, so
threads and processes need no lock, and a slot written twice at once keeps the
last write. The 256 shard directories keep each directory small (a million
completions is ~4,000 files per directory). JSON, unlike the op cache's pickle,
cannot execute code when read.

Beside the slots, ``requests/rows/<id[:2]>/<id>.json`` remembers, for each
(op, step, row), the key of the last request that answered it and what that
cost. The id leaves out the prompt, so a batch can see that a row is asked
again only because its prompt changed (see ``resent_warning``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pyforger._render import fmt_cost, fmt_units
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
    if not isinstance(entry, dict) or not (
        isinstance(entry.get("text"), str) or is_row_error(entry)
    ):
        return None
    return entry


def row_error_entry(reason: str, error: BaseException) -> dict[str, Any]:
    """What a slot stores for a request that hit a row error.

    E.g. ``{"row_error": "content filter", "error": "BadRequestError: ..."}``.
    The error repeats, or may not, on a rerun; either way the rerun should get
    the rows the first run got, so it replays the error (see Model).
    """
    text = " ".join(f"{type(error).__name__}: {error}".split())
    return {"row_error": reason, "error": text[:500]}


def is_row_error(entry: dict[str, Any]) -> bool:
    """True for a slot that stores a row error, not a completion."""
    return isinstance(entry.get("row_error"), str)


def write_slot(key: str, slot: int, entry: dict[str, Any]) -> None:
    """Store one completion atomically; readers never see a partial file."""
    path = _slot_path(key, slot)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(entry).encode())


# Rows asked again under a changed prompt warn only when there are this many,
# and at least this share of the batch: editing a few rows' prompts is routine.
MIN_RESENT_ROWS = 20
MIN_RESENT_SHARE = 0.5


def row_id(op_name: str, step: str | None, row: str) -> str:
    """Hex id of a row's "answered before" entry: op, step, and row, no prompt.

    e.g. ("alter.generate_from_context", "write_answers", <row hash>) stays the
    same when the step's template changes, while the request key does not.
    """
    key, _ = _hash([_KEY_VERSION, "row", op_name, step, row])
    return key


def _row_path(row_id: str) -> Path:
    return _cache_dir() / "requests" / "rows" / row_id[:2] / f"{row_id}.json"


def read_row(row_id: str) -> dict[str, Any] | None:
    """The row's entry, e.g. {"key": ..., "run": ..., "units": 5e4}, or None."""
    try:
        with _row_path(row_id).open("rb") as f:
            if hasattr(os, "getuid") and os.fstat(f.fileno()).st_uid != os.getuid():
                return None
            entry = json.loads(f.read())
    except Exception:  # absent or unreadable: nothing to warn about
        return None
    if not isinstance(entry, dict) or not isinstance(entry.get("key"), str):
        return None
    return entry


def write_row(row_id: str, entry: dict[str, Any]) -> None:
    """Replace the row's entry atomically (the last answer wins)."""
    path = _row_path(row_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(entry).encode())


def resent_warning(
    op_name: str, earlier: list[dict[str, Any]], total: int
) -> str | None:
    """The warning for rows answered before under another prompt, or None.

    `earlier` holds the row entries of the batch's rows that will be asked
    again; `total` is the batch's row count. e.g. 249 of 498 rows give
    "alter.generate_from_context: 249 of 498 rows were answered before under a
    different prompt or settings and will be asked again (the request cache
    keys on the exact prompt); ~12.45M token units at the earlier cost per
    row". Under MIN_RESENT_ROWS rows or MIN_RESENT_SHARE of the batch: None.
    """
    count = len(earlier)
    if count < MIN_RESENT_ROWS or count < MIN_RESENT_SHARE * total:
        return None
    text = (
        f"{op_name}: {count} of {total} rows were answered before under a "
        "different prompt or settings and will be asked again (the request "
        "cache keys on the exact prompt)"
    )
    units = _total(e.get("units") for e in earlier)
    cost = _total(e.get("cost_usd") for e in earlier)
    if units:
        text += f"; ~{fmt_units(units)} token units at the earlier cost per row"
    elif cost:
        text += f"; ~{fmt_cost(cost)} at the earlier cost per row"
    return text


def _total(values: Iterable[Any]) -> float | None:
    """Sum of `values`; None when any is unknown (not a number)."""
    total = 0.0
    for value in values:
        if not isinstance(value, (int, float)):
            return None
        total += value
    return total
