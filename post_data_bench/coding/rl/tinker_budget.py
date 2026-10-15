"""Dollar ledgers for Tinker calls: SFT ($150) and RL ($400) per agent run.

Charges come from Tinker's own per-call token counts at Tinker's rate card,
including prompt prefix-cache hits reported by each sample response. The
billing API (`reconcile`) lags by hours, so it audits ledgers after a run but
cannot gate calls. Standard library only, so agents can import it directly.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import tempfile
import threading
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3-8B"
# USD per million tokens, from the Tinker pricing page (2026-10).
PRICES_PER_MILLION = {
    "Qwen/Qwen3-8B": {
        "prefill": 0.195,
        "cached_prefill": 0.039,
        "sample": 0.60,
        "train": 0.44,
    },
}
DEFAULT_LIMITS_USD = {"sft": 150.0, "rl": 400.0}
# Tinker allows ~2000 in-flight sample requests per account; a lower per-process
# cap leaves room for several agents running in parallel.
DEFAULT_MAX_CONCURRENT_SAMPLES = 600
# eval: host-side secret-test evaluation; recorded, never limited.
BUDGETS = ("sft", "rl", "eval")

# Worst-case dollars of this process's in-flight calls, per ledger file, so
# concurrent calls cannot each pass the limit check on the same spend.
_RESERVED: dict[Path, tuple[int, float]] = {}  # path -> (calls, dollars)
_RESERVED_LOCK = threading.Lock()
# (event loop, future) of calls waiting for an in-flight call to finish.
_WAITERS: list[tuple[asyncio.AbstractEventLoop, asyncio.Future]] = []


class TinkerBudgetExceeded(RuntimeError):
    """Raised before a Tinker call that would cross the run's dollar limit."""


def prices(model):
    """Return the rate card for a base model; unknown models cannot be budgeted."""
    try:
        return PRICES_PER_MILLION[model]
    except KeyError:
        raise ValueError(f"No Tinker prices for {model}; add them first") from None


def cost_usd(model, tokens):
    """Price a {kind: token_count} mapping, e.g. {"train": 8192}."""
    rates = prices(model)
    return sum(rates[kind] * count for kind, count in tokens.items()) / 1e6


def sample_tokens(prompt_length, num_samples, cache_hit_tokens, sampled_lengths):
    """Billed token kinds for one sample request.

    Tinker bills the shared prompt once at prefill rates, minus its reported
    prefix-cache hits; the other num_samples - 1 copies are cache hits.
    """
    hits = min(cache_hit_tokens, prompt_length)
    return {
        "prefill": prompt_length - hits,
        "cached_prefill": hits + prompt_length * (num_samples - 1),
        "sample": sum(sampled_lengths),
    }


def max_concurrent_samples():
    """In-flight Tinker sample requests per process.

    Set by TINKER_MAX_CONCURRENT_SAMPLES.
    """
    value = int(
        os.environ.get("TINKER_MAX_CONCURRENT_SAMPLES", DEFAULT_MAX_CONCURRENT_SAMPLES)
    )
    if value < 1:
        raise ValueError("TINKER_MAX_CONCURRENT_SAMPLES must be positive")
    return value


def budget_from_env(name):
    """Open the ledger the launcher configured for `name` (sft, rl, or eval)."""
    if name not in BUDGETS:
        raise ValueError(f"budget must be one of {BUDGETS}")
    prefix = f"TINKER_{name.upper()}"
    path = os.environ.get(prefix + "_USAGE_FILE")
    if not path:
        raise ValueError(f"Set {prefix}_USAGE_FILE to this run's ledger")
    limit = os.environ.get(prefix + "_USD_LIMIT", DEFAULT_LIMITS_USD.get(name))
    return TinkerBudget(path, limit, name=name)


def _wake(waiter):
    if not waiter.done():
        waiter.set_result(None)


class TinkerBudget:
    """Share Tinker spend across processes; refuse calls that would exceed the limit.

    `limit=None` records without limiting. Calls made through `reserve` in one
    process never jointly overshoot the limit; calls in other processes, or
    checked with `check` alone, can overshoot it by at most their own in-flight
    cost. Needs POSIX locks (not GCS FUSE).
    """

    def __init__(self, path, limit, name="sft", model=BASE_MODEL):
        """Bind a ledger file to a positive dollar limit (or None)."""
        if limit is not None:
            limit = float(limit)
            if not math.isfinite(limit) or limit <= 0:
                raise ValueError("Tinker dollar limit must be positive and finite")
        self.limit, self.name, self.model = limit, name, model
        prices(model)
        self.path = Path(path).absolute()

    @contextlib.contextmanager
    def _locked(self):
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            str(self.path) + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
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
                "budget": self.name,
                "model": self.model,
                "limit_usd": self.limit,
                "cost_usd": 0.0,
                "calls": 0,
                "tokens": {kind: 0 for kind in prices(self.model)},
                "sessions": [],
            }
        with os.fdopen(descriptor) as stream:
            state = json.load(stream)
        if (state.get("version"), state.get("limit_usd"), state.get("model")) != (
            1,
            self.limit,
            self.model,
        ):
            raise ValueError("Tinker ledger limit/model differs from this run")
        if not math.isfinite(state["cost_usd"]) or state["cost_usd"] < 0:
            raise ValueError("Invalid Tinker ledger")
        return state

    def _write(self, state):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.path.parent, prefix=".tinker-usage-", delete=False
            ) as stream:
                temporary = stream.name
                json.dump(self._snapshot(state), stream, sort_keys=True, indent=1)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def _snapshot(self, state):
        remaining = (
            None if self.limit is None else max(0.0, self.limit - state["cost_usd"])
        )
        return {**state, "remaining_usd": remaining}

    def snapshot(self):
        """Read spend, creating an empty ledger on first use."""
        with self._locked():
            state = self._read()
            if not self.path.exists():
                self._write(state)
            return self._snapshot(state)

    def check(self, estimate_usd=0.0):
        """Raise unless spend plus this call's estimated cost stays within the limit."""
        state = self.snapshot()
        if self.limit is not None and state["cost_usd"] + estimate_usd > self.limit:
            raise TinkerBudgetExceeded(
                f"Tinker {self.name} budget: ${state['cost_usd']:.2f} spent of "
                f"${self.limit:.2f}; next call needs up to ${estimate_usd:.2f}"
            )
        return state

    @contextlib.asynccontextmanager
    async def reserve(self, estimate_usd):
        """Hold a call's worst-case cost against the limit while it runs.

        `async with budget.reserve(cost): ...` checks spend plus this process's
        other reservations on the same ledger, so e.g. 600 concurrent requests
        with $1 left cannot all start. When the call does not fit but others
        are in flight, it waits (without blocking the event loop) for them to
        finish; when none are, it raises `TinkerBudgetExceeded`. Record the
        call's tokens before leaving the block. The reservation is released on
        exit, including on errors and cancellation.
        """
        loop = asyncio.get_running_loop()
        while True:
            with _RESERVED_LOCK:
                count, reserved = _RESERVED.get(self.path, (0, 0.0))
                try:
                    self.check(reserved + estimate_usd)
                except TinkerBudgetExceeded:
                    if not count:
                        raise
                    # Registered under the lock, so the next release wakes it.
                    waiter = loop.create_future()
                    _WAITERS.append((loop, waiter))
                else:
                    _RESERVED[self.path] = (count + 1, reserved + estimate_usd)
                    break
            await waiter
        try:
            yield
        finally:
            with _RESERVED_LOCK:
                count, reserved = _RESERVED.pop(self.path)
                if count > 1:
                    _RESERVED[self.path] = (count - 1, reserved - estimate_usd)
                waiters = list(_WAITERS)
                _WAITERS.clear()
            for waiter_loop, waiter in waiters:
                with contextlib.suppress(RuntimeError):  # its loop has closed
                    waiter_loop.call_soon_threadsafe(_wake, waiter)

    def record(self, tokens, session_id=None):
        """Charge one call's billed tokens, e.g. {"train": n} or sample_tokens(...)."""
        cost = cost_usd(self.model, tokens)
        with self._locked():
            state = self._read()
            for kind, count in tokens.items():
                state["tokens"][kind] += int(count)
            state["cost_usd"] += cost
            state["calls"] += 1
            if session_id and session_id not in state["sessions"]:
                state["sessions"].append(session_id)
            self._write(state)
            return self._snapshot(state)


def reconcile(ledger_path, starting_on, ending_before):
    """Compare a ledger with Tinker's billing records for the sessions it lists.

    Billing data lags by up to a few hours and needs billing-view access.
    Window bounds are RFC 3339 hours, e.g. "2026-10-13T00:00:00Z".
    """
    import tinker

    ledger = json.loads(Path(ledger_path).read_text())
    sessions = set(ledger["sessions"])
    rest = tinker.ServiceClient().create_rest_client()
    billed = {"cost_usd": 0.0, "unpriced_events": 0, "tokens": {}}
    for event in rest.get_billing_usage(starting_on, ending_before).result().data:
        if event.session_id not in sessions:
            continue
        info = event.event_info
        count = getattr(info, "token_count", None)
        if count is not None:
            kind = info.type
            if getattr(info, "cached", False):
                kind += "_cached"
            billed["tokens"][kind] = billed["tokens"].get(kind, 0) + count
        if count is None:
            continue  # checkpoint/storage events are not charged to ledgers
        cost = event.estimated_cost_usd
        if cost is None:
            billed["unpriced_events"] += 1
        else:
            billed["cost_usd"] += cost
    return {
        "ledger_cost_usd": ledger["cost_usd"],
        "ledger_tokens": ledger["tokens"],
        "billed": billed,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger")
    parser.add_argument(
        "--reconcile", nargs=2, metavar=("STARTING_ON", "ENDING_BEFORE")
    )
    options = parser.parse_args()
    if options.reconcile:
        print(json.dumps(reconcile(options.ledger, *options.reconcile), indent=2))
    else:
        print(Path(options.ledger).read_text())
