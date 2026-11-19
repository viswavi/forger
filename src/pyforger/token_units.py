"""Token-unit weights you choose, and a process-safe, persistent usage ledger.

Forger always records dollars. Token units are an optional second measure: a
weighted token count, e.g. a model's prices in units of the cheapest price.
No model has weights until you give it some, with `set_weights` or the
``FORGER_TOKEN_UNIT_WEIGHTS`` environment variable; until then a model's
units are NaN and reports show dollars only.
"""

from __future__ import annotations

import contextlib
import functools
import json
import math
import os
import socket
import tempfile
import time
import uuid
from collections.abc import Mapping
from pathlib import Path

WEIGHTS_ENV = "FORGER_TOKEN_UNIT_WEIGHTS"
# Weights given with set_weights; None means "read WEIGHTS_ENV".
_set_weights: dict[str, tuple[float, float]] | None = None
# Completion tokens held for a request sent without max_tokens.
_DEFAULT_HELD_COMPLETION = 65_536
# Prompt characters per held token: below the 3-4 real tokenizers give.
_HELD_CHARS_PER_TOKEN = 3
# A hold this old belongs to a killed process: no attempt runs this long. It is
# released, like the hold of a process seen to have died.
_STALE_HOLD_S = 2 * 60 * 60
# Statuses the gateway returns without running the model, so nothing is billed:
# bad request, auth, forbidden, not found, payload too large, unprocessable, 429,
# and 502/503, which a proxy in front of the gateway returns in ~0.1 s while the
# gateway is down (2026-11-01). 500 and 504 stay billed: the model may have run.
_UNBILLED_STATUS = frozenset({400, 401, 403, 404, 413, 422, 429, 502, 503})


class TokenUnitQuotaExceeded(RuntimeError):
    """No more calls may start under this token-unit quota."""


def _parse_weights(table, source):
    """``{"name": (input, output)}`` with lowercase names and float weights."""
    if not isinstance(table, Mapping):
        raise ValueError(f"{source} must map model names to [input, output] weights")
    parsed = {}
    for name, pair in table.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{source}: model names must be nonempty strings")
        try:
            if isinstance(pair, (str, bytes)) or isinstance(pair, Mapping):
                raise TypeError
            input_weight, output_weight = (float(w) for w in pair)
        except (TypeError, ValueError):
            raise ValueError(
                f"{source}: {name!r} needs two weights, [input, output]"
            ) from None
        if not all(math.isfinite(w) and w >= 0 for w in (input_weight, output_weight)):
            raise ValueError(f"{source}: {name!r} weights must be finite and >= 0")
        parsed[name.strip().lower()] = (input_weight, output_weight)
    return parsed


def set_weights(weights):
    """Give models token-unit weights, or ``None`` to go back to the environment.

    ``weights`` maps part of a model id (any case) to its ``(input, output)``
    units per token, e.g. ``{"claude-haiku": (1, 5), "claude-sonnet": (3, 15)}``.
    A model id takes the weights of the longest name it contains. Models no
    name matches have no units (NaN). This replaces ``FORGER_TOKEN_UNIT_WEIGHTS``
    in this process only; set that variable (a JSON object of the same form)
    for weights that other processes sharing a ledger also see.
    """
    global _set_weights
    _set_weights = None if weights is None else _parse_weights(weights, "set_weights")


@functools.lru_cache(maxsize=8)
def _env_weights(raw):
    try:
        table = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(
            f"{WEIGHTS_ENV} must be a JSON object, e.g. " '{"claude-haiku": [1, 5]}'
        ) from None
    return _parse_weights(table, WEIGHTS_ENV)


def weights():
    """The weights in force: `set_weights`'s, else ``FORGER_TOKEN_UNIT_WEIGHTS``'s.

    Empty unless you gave some.
    """
    if _set_weights is not None:
        return dict(_set_weights)
    raw = os.environ.get(WEIGHTS_ENV, "").strip()
    return dict(_env_weights(raw)) if raw else {}


def model_weights(model_id):
    """Return the ``(input, output)`` weights for ``model_id``.

    Raises ValueError when no weights name matches it, or when two names of the
    same length match with different weights.
    """
    if not isinstance(model_id, str):
        raise ValueError("model_id must be a string")
    lowered = model_id.lower()
    matches = [(name, w) for name, w in weights().items() if name in lowered]
    if not matches:
        raise ValueError(
            f"No token-unit weights for {model_id!r}; give them with "
            f"pyforger.token_units.set_weights or {WEIGHTS_ENV}"
        )
    longest = max(len(name) for name, _ in matches)
    best = {w for name, w in matches if len(name) == longest}
    if len(best) != 1:
        raise ValueError(f"Token-unit weights for {model_id!r} are ambiguous")
    return best.pop()


def has_weights(model_id):
    """Whether ``model_id`` has token-unit weights."""
    try:
        model_weights(model_id)
    except ValueError:
        return False
    return True


def _count(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Token counts must be nonnegative integers")
    return value


def token_units(model_id, prompt_tokens, completion_tokens):
    """Weight all input tokens and output tokens, including reasoning once.

    Raises ValueError for a model without weights (see `set_weights`).
    """
    input_weight, output_weight = model_weights(model_id)
    return float(
        _count(prompt_tokens) * input_weight + _count(completion_tokens) * output_weight
    )


def response_token_counts(response):
    """Read required aggregate usage from a LiteLLM response; never guess zero."""

    def field(obj, name):
        return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)

    usage = field(response, "usage")
    return _count(field(usage, "prompt_tokens")), _count(
        field(usage, "completion_tokens")
    )


def _valid_cost(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if math.isfinite(value) and value >= 0 else None
    except (OverflowError, ValueError):
        return None


def response_cost_usd(response, model_id):
    """Return available USD pricing without making token metering depend on it."""
    try:
        hidden = (
            response.get("_hidden_params")
            if isinstance(response, dict)
            else getattr(response, "_hidden_params", None)
        )
        cost = (
            _valid_cost(hidden.get("response_cost"))
            if isinstance(hidden, dict)
            else None
        )
        if cost is not None:
            return cost
    except Exception:
        pass
    try:
        import litellm

        return _valid_cost(
            litellm.completion_cost(completion_response=response, model=model_id)
        )
    except Exception:
        return None


def _normalize_cost_metadata(state):
    """Treat missing or damaged cost metadata as incomplete, preserving tokens."""
    cost = _valid_cost(state.get("cost_usd"))
    unpriced = state.get("unpriced_calls")
    changed = cost is None
    if cost is None:
        cost, unpriced = 0.0, state["calls"]
    elif (
        isinstance(unpriced, bool)
        or not isinstance(unpriced, int)
        or not 0 <= unpriced <= state["calls"]
    ):
        unpriced = state["calls"]
        changed = True
    state["cost_usd"], state["unpriced_calls"] = cost, unpriced
    return changed


def held_units(model_id, messages, n=1, max_tokens=None):
    """Worst-case units of one request: its prompt and ``n`` full completions.

    The prompt counts 1 token per 3 characters (more than real tokenizers
    give); completions count ``max_tokens`` each, or 65,536 without one.
    """
    chars = sum(
        len(str(m.get("content", ""))) if isinstance(m, dict) else len(str(m))
        for m in messages
    )
    completion = max_tokens or _DEFAULT_HELD_COMPLETION
    return token_units(model_id, -(-chars // _HELD_CHARS_PER_TOKEN), n * completion)


def billed_on_error(exc):
    """Whether a failed request may still be billed, e.g. after a timeout.

    The gateway finishes (and bills) a request whose client timed out or lost
    the connection; it bills nothing it refused (4xx). A process that is
    interrupted or exits mid-request releases its holds instead (see
    `TokenUnitBudget`).
    """
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return False
    return getattr(exc, "status_code", None) not in _UNBILLED_STATUS


def _owner():
    """This process's identity for its holds: pid, host and pid namespace."""
    try:
        namespace = os.readlink("/proc/self/ns/pid")
    except OSError:
        namespace = None
    return {"pid": os.getpid(), "host": socket.gethostname(), "pid_ns": namespace}


def _owner_dead(entry, me):
    """Whether a hold's process is known to be gone: same host and namespace only."""
    if entry.get("host") != me["host"] or entry.get("pid_ns") != me["pid_ns"]:
        return False
    pid = entry.get("pid")
    if not isinstance(pid, int) or pid <= 0 or pid == me["pid"]:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False  # alive but not ours to signal
    return False


class TokenUnitBudget:
    """Share actual usage across processes; gate new calls once the quota is used.

    Each request holds its worst case (`held_units`) while in flight, and the
    hold counts as used: a response replaces it with actual usage, and a
    failure that may be billed (timeout, lost connection) keeps it. A process
    that dies mid-request releases its holds rather than charging them: the
    next read of the ledger drops holds whose process is gone (same host and
    pid namespace), or any hold two hours old, and counts them in
    `released_calls`; an interrupt or exit inside a request releases its hold
    too. Requests the gateway still finishes for a killed process can then
    overshoot the quota slightly.
    A separate lock file protects atomic ledger replacement. Use a shared
    filesystem with POSIX file locks.
    """

    def __init__(self, path, limit):
        """Bind a ledger path to a positive quota without opening resources."""
        if isinstance(limit, bool):
            raise ValueError("Token-unit limit must be positive and finite")
        try:
            self.limit = float(limit)
        except (TypeError, ValueError):
            raise ValueError("Token-unit limit must be positive and finite") from None
        if not math.isfinite(self.limit) or self.limit <= 0:
            raise ValueError("Token-unit limit must be positive and finite")
        if not path:
            raise ValueError("Token usage file is required")
        self.path = Path(path).absolute()

    @contextlib.contextmanager
    def _locked(self):
        # Ordinary Model accounting also works on platforms without POSIX locks.
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = str(self.path) + ".lock"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _read(self):
        try:
            descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return {
                "version": 1,
                "limit": self.limit,
                "token_units": 0.0,
                "cost_usd": 0.0,
                "unpriced_calls": 0,
                "calls": 0,
                "models": {},
                "unmetered_calls": 0,
                "held": {},
                "held_token_units": 0.0,
                "unreturned_calls": 0,
                "unreturned_token_units": 0.0,
                "released_calls": 0,
                "released_token_units": 0.0,
            }
        with os.fdopen(descriptor) as stream:
            state = json.load(stream)
        if state.get("version") != 1 or state.get("limit") != self.limit:
            raise ValueError("Token ledger policy/quota differs from this run")
        used = state.get("token_units")
        if (
            isinstance(used, bool)
            or not isinstance(used, (int, float))
            or not math.isfinite(used)
            or used < 0
        ):
            raise ValueError("Invalid token usage ledger")
        _count(state.get("calls"))
        _count(state.get("unmetered_calls"))
        if not isinstance(state.get("models"), dict):
            raise ValueError("Invalid token usage ledger")
        # Ledgers written before holds existed have none.
        state.setdefault("held", {})
        state.setdefault("held_token_units", 0.0)
        state.setdefault("unreturned_calls", 0)
        state.setdefault("unreturned_token_units", 0.0)
        state.setdefault("released_calls", 0)
        state.setdefault("released_token_units", 0.0)
        if not isinstance(state["held"], dict):
            raise ValueError("Invalid token usage ledger")
        me = _owner()
        orphaned = [
            hold
            for hold, entry in state["held"].items()
            if time.time() - entry.get("since", 0) > _STALE_HOLD_S
            or _owner_dead(entry, me)
        ]
        for hold in orphaned:
            self._release_hold(state, hold)
        changed = bool(orphaned)
        changed = _normalize_cost_metadata(state) or changed
        for model in state["models"].values():
            changed = _normalize_cost_metadata(model) or changed
        if changed:
            self._write(state)
        return state

    def _write(self, state):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.path.parent, prefix=".token-usage-", delete=False
            ) as stream:
                temporary = stream.name
                json.dump(self._snapshot(state), stream, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def _snapshot(self, state):
        return {
            **state,
            "remaining_token_units": max(
                0.0, self.limit - state["token_units"] - state["held_token_units"]
            ),
        }

    def snapshot(self):
        """Read usage, initializing an empty ledger for a new run."""
        with self._locked():
            state = self._read()
            if not self.path.exists():
                self._write(state)
            return self._snapshot(state)

    def check(self):
        """Reject new calls when usage is incomplete or the quota is exhausted."""
        state = self.snapshot()
        if state["unmetered_calls"]:
            raise ValueError("Token accounting is incomplete: a response omitted usage")
        if state["token_units"] + state["held_token_units"] >= self.limit:
            raise TokenUnitQuotaExceeded("Token-unit quota exhausted")
        return state

    @contextlib.contextmanager
    def hold(self, model_id, units):
        """Hold ``units`` for one request sent inside the block; yield the hold's id.

        Pass the id to `record` (or `mark_unmetered`) once the response
        arrives. If the block raises, the hold is charged when the request
        may have been billed (`billed_on_error`) and dropped otherwise; an
        interrupt or exit releases it (`released_calls`).
        """
        hold = f"{os.getpid()}-{uuid.uuid4().hex}"
        with self._locked():
            state = self._read()
            state["held"][hold] = {
                "model": model_id,
                "token_units": float(units),
                "since": time.time(),
                **_owner(),
            }
            state["held_token_units"] += units
            self._write(state)
        try:
            yield hold
        except BaseException as exc:
            with self._locked():
                state = self._read()
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    self._release_hold(state, hold)
                elif billed_on_error(exc):
                    self._charge_hold(state, hold)
                else:
                    self._drop_hold(state, hold)
                self._write(state)
            raise

    def _release_hold(self, state, hold):
        """Drop a hold whose process stopped mid-request, and count it."""
        entry = self._drop_hold(state, hold)
        if entry is None:
            return
        state["released_calls"] += 1
        state["released_token_units"] += entry["token_units"]

    def _drop_hold(self, state, hold):
        entry = state["held"].pop(hold, None)
        if entry is None:
            return None
        state["held_token_units"] = max(
            0.0, state["held_token_units"] - entry["token_units"]
        )
        return entry

    def _charge_hold(self, state, hold):
        """Count a hold as spent: its request may be billed but never reported."""
        entry = self._drop_hold(state, hold)
        if entry is None:
            return
        units = entry["token_units"]
        model = self._model_entry(state, entry["model"])
        model["token_units"] += units
        model["unreturned_calls"] = model.get("unreturned_calls", 0) + 1
        state["token_units"] += units
        state["unreturned_calls"] += 1
        state["unreturned_token_units"] += units

    @staticmethod
    def _model_entry(state, model_id):
        return state["models"].setdefault(
            model_id,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "token_units": 0.0,
                "calls": 0,
                "cost_usd": 0.0,
                "unpriced_calls": 0,
            },
        )

    def mark_unmetered(self, hold=None):
        """Fail closed after a paid response without usable token counts.

        Its ``hold``, if any, is charged.
        """
        with self._locked():
            state = self._read()
            if hold is not None:
                self._charge_hold(state, hold)
            state["unmetered_calls"] += 1
            self._write(state)

    def record(
        self,
        model_id,
        prompt_tokens,
        completion_tokens,
        *,
        cost_usd=None,
        hold=None,
    ):
        """Charge tokens and available dollars, replacing the request's ``hold``."""
        units = token_units(model_id, prompt_tokens, completion_tokens)
        cost = _valid_cost(cost_usd)
        with self._locked():
            state = self._read()
            if hold is not None:
                self._drop_hold(state, hold)
            model = self._model_entry(state, model_id)
            model["prompt_tokens"] += prompt_tokens
            model["completion_tokens"] += completion_tokens
            model["token_units"] += units
            model["calls"] += 1
            state["token_units"] += units
            state["calls"] += 1
            if cost is None or any(
                _valid_cost(entry["cost_usd"] + cost) is None
                for entry in (state, model)
            ):
                model["unpriced_calls"] += 1
                state["unpriced_calls"] += 1
            else:
                model["cost_usd"] += cost
                state["cost_usd"] += cost
            self._write(state)
            return self._snapshot(state)
