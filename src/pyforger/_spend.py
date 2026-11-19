"""The spend ledger: one JSON line per LLM result, written as it is recorded.

``Model``, ``TinkerModel``, ``LocalModel``, and ``dynamic_step`` agents add
every result's usage through ``_tracking.add_usage``, which appends one line
here right away, so a run that is killed or raises keeps the record of every
result recorded before it stopped (``Model`` and ``TinkerModel`` record each
request as it finishes, ``LocalModel`` each batch when it returns)::

    {"timestamp": "2026-10-16T18:02:11.204+00:00", "pid": 4121,
     "run_id": "3f9c0a7e1b2d", "pipeline": "main", "step": "write_solutions",
     "op": "alter.generate_from_context", "model_id": "azure_ai/gpt-5.6-luna",
     "prompt_tokens": 812, "completion_tokens": 240, "cost_usd": 0.0031,
     "token_units": null}

``token_units`` is null for a model without token-unit weights (see
``pyforger.token_units``).

The file is ``$FORGER_SPEND_LEDGER`` if set, else ``$FORGER_CACHE/spend.jsonl``
(cache dir default ``~/.cache/forger``), both read at call time.
``FORGER_SPEND_LEDGER=0`` turns the ledger off. ``pyforger.inspect.spend``
totals it.

This module imports nothing from the rest of pyforger.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # Windows: the thread lock still keeps lines whole
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_OFF = {"0", "false", "no", "off"}
_LOCK = threading.Lock()
_RUN: dict[str, Any] = {"pid": None, "id": ""}
_FAILED_PATHS: set[str] = set()  # warn once per unwritable ledger


def run_id() -> str:
    """Return this process's run id (a new one after a fork)."""
    with _LOCK:
        if _RUN["pid"] != os.getpid():
            _RUN["pid"] = os.getpid()
            _RUN["id"] = uuid.uuid4().hex[:12]
        return str(_RUN["id"])


def ledger_path() -> Path | None:
    """Return the ledger file, or None when ``FORGER_SPEND_LEDGER=0``."""
    raw = os.environ.get("FORGER_SPEND_LEDGER", "").strip()
    if raw.lower() in _OFF:
        return None
    if raw:
        return Path(raw).expanduser()
    cache = os.environ.get("FORGER_CACHE", Path.home() / ".cache" / "forger")
    return Path(cache) / "spend.jsonl"


def append(
    *,
    pipeline: str | None,
    step: str | None,
    op: str | None,
    model_id: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    token_units: float | None = None,
) -> None:
    """Append one usage record to the ledger; never raises.

    ``token_units`` is None for a model without token-unit weights.

    Each line is written under a lock (also across processes where ``fcntl``
    exists) and flushed before the call returns.
    """
    path = ledger_path()
    if path is None:
        return
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "pid": os.getpid(),
        "run_id": run_id(),
        "pipeline": pipeline,
        "step": step,
        "op": op,
        "model_id": model_id,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": cost_usd,
        "token_units": token_units,
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, path.open("a", encoding="utf-8") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError as exc:  # spend accounting must never fail a run
        if str(path) not in _FAILED_PATHS:
            _FAILED_PATHS.add(str(path))
            logger.warning("cannot write the spend ledger %s: %s", path, exc)


def parse_time(value: datetime | str) -> datetime:
    """Parse an ISO timestamp; a time without a zone is local time."""
    when = datetime.fromisoformat(value) if isinstance(value, str) else value
    return when if when.tzinfo is not None else when.astimezone()


def read(path: Path, since: datetime | None = None) -> list[dict[str, Any]]:
    """Return the ledger's records, oldest first, skipping unreadable lines.

    A run killed mid-write can leave a partial last line; it is skipped.
    """
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                when = parse_time(record["timestamp"])
            except (ValueError, KeyError, TypeError):
                continue
            if since is None or when >= since:
                records.append(record)
    return records
