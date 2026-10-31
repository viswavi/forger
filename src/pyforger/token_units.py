"""Fixed token-unit weights and a process-safe, persistent usage ledger."""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
import uuid
from pathlib import Path

DEFAULT_GENERATION_LIMIT = 180_000_000
DEFAULT_JUDGE_MODEL = "litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731"
DEFAULT_REASONING_EFFORT = "low"
_WEIGHTS = {"deepseek-v4-flash": (1, 3), "glm-5.3": (4, 12.5)}
# Completion tokens held for a request sent without max_tokens.
_DEFAULT_HELD_COMPLETION = 65_536
# Statuses the gateway returns without running the model, so nothing is billed:
# bad request, auth, forbidden, not found, payload too large, unprocessable, 429.
_UNBILLED_STATUS = frozenset({400, 401, 403, 404, 413, 422, 429})


class TokenUnitQuotaExceeded(RuntimeError):
    """No more calls may start under this token-unit quota."""


def model_weights(model_id):
    """Return fixed input/output weights for a supported model or alias."""
    if not isinstance(model_id, str):
        raise ValueError("model_id must be a string")
    matches = [
        weights for name, weights in _WEIGHTS.items() if name in model_id.lower()
    ]
    if len(matches) != 1:
        raise ValueError(
            "Token units require a DeepSeek-V4-Flash or GLM-5.3 model alias"
        )
    return matches[0]


def _count(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Token counts must be nonnegative integers")
    return value


def token_units(model_id, prompt_tokens, completion_tokens):
    """Weight all input tokens and output tokens, including reasoning once."""
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


def validate_judge_model(model):
    """Accept only aliases identifying DeepSeek-V4-Flash."""
    if not isinstance(model, str) or "deepseek-v4-flash" not in model.lower():
        raise ValueError("JUDGE_MODEL must contain DeepSeek-V4-Flash")
    model_weights(model)
    return model


def validate_reasoning_effort(effort):
    """Validate the supported DeepSeek judge reasoning levels."""
    if effort not in ("none", "low", "high"):
        raise ValueError("JUDGE_REASONING_EFFORT must be none, low, or high")
    return effort


def default_generation_limit(task):
    """Return the generation quota for a launcher task name.

    The algorithmic task recommends ~500 rows (800 at most) instead of ~1,250,
    so its quota shrinks to match.
    """
    normalized = str(task).strip().lower()
    if normalized in ("coding", "culture"):
        return DEFAULT_GENERATION_LIMIT
    if normalized == "algorithmic":
        return 120_000_000
    raise ValueError("Unknown task for the generation token-unit quota")


def default_rl_limit(task):
    """Return the fixed RL quota for a launcher task name.

    Only task names appear here: this module is copied into agent sandboxes,
    which must never see benchmark names.
    """
    normalized = str(task).strip().lower()
    if normalized in ("coding", "culture"):
        return 50_000_000
    if normalized == "algorithmic":
        # Responses run ~15K tokens, so judging them costs more per rollout.
        return 150_000_000
    raise ValueError("Unknown task for RL token-unit quota")


def held_units(model_id, messages, n=1, max_tokens=None):
    """Worst-case units of one request: its prompt and ``n`` full completions.

    The prompt counts 1 token per 2 characters (more than real tokenizers
    give); completions count ``max_tokens`` each, or 65,536 without one.
    """
    chars = sum(
        len(str(m.get("content", ""))) if isinstance(m, dict) else len(str(m))
        for m in messages
    )
    completion = max_tokens or _DEFAULT_HELD_COMPLETION
    return token_units(model_id, -(-chars // 2), n * completion)


def billed_on_error(exc):
    """Whether a failed request may still be billed, e.g. after a timeout.

    The gateway finishes (and bills) a request whose client timed out, lost
    the connection, or was interrupted; it bills nothing it refused (4xx).
    """
    return getattr(exc, "status_code", None) not in _UNBILLED_STATUS


class TokenUnitBudget:
    """Share actual usage across processes; gate new calls once the quota is used.

    Each request holds its worst case (`held_units`) while in flight, and the
    hold counts as used: a response replaces it with actual usage, a failure
    that may be billed (timeout, lost connection, killed process) keeps it.
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
        if not isinstance(state["held"], dict):
            raise ValueError("Invalid token usage ledger")
        changed = _normalize_cost_metadata(state)
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
        may have been billed (`billed_on_error`) and dropped otherwise.
        """
        hold = f"{os.getpid()}-{uuid.uuid4().hex}"
        with self._locked():
            state = self._read()
            state["held"][hold] = {"model": model_id, "token_units": float(units)}
            state["held_token_units"] += units
            self._write(state)
        try:
            yield hold
        except BaseException as exc:
            with self._locked():
                state = self._read()
                if billed_on_error(exc):
                    self._charge_hold(state, hold)
                else:
                    self._drop_hold(state, hold)
                self._write(state)
            raise

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
        self, model_id, prompt_tokens, completion_tokens, *, cost_usd=None, hold=None
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
