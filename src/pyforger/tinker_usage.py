"""Tinker prices and a process-safe dollar ledger for Tinker sampling.

Standard library only. Other processes that write the same file format and lock
(e.g. SFT or evaluation jobs) can share one capped ledger file with `TinkerModel`.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

# USD per million tokens, from the Tinker pricing page (2026-10). "cached_prefill"
# is a prompt token served from the prefix cache; "train" is listed so new ledgers
# have the same token kinds as training ledgers.
TINKER_PRICES_PER_MILLION: dict[str, dict[str, float]] = {
    "Qwen/Qwen3-8B": {
        "prefill": 0.195,
        "cached_prefill": 0.039,
        "sample": 0.60,
        "train": 0.44,
    },
}
# Tinker allows ~2000 in-flight sample requests per account; a lower per-process
# cap leaves room for several processes running in parallel.
DEFAULT_MAX_CONCURRENT_SAMPLES = 600

# Worst-case dollars of this process's in-flight calls, per ledger file, so
# concurrent calls cannot each pass the limit check on the same spend.
_RESERVED: dict[Path, tuple[int, float]] = {}
_RESERVED_CHANGED = threading.Condition()


class TinkerBudgetExceeded(RuntimeError):
    """Raised before a Tinker call that would cross the ledger's dollar limit."""


def tinker_prices(model: str) -> dict[str, float]:
    """Return the built-in rate card for a base model; unknown models raise."""
    try:
        return TINKER_PRICES_PER_MILLION[model]
    except KeyError:
        raise ValueError(
            f"No Tinker prices for {model}; pass prices= to TinkerModel"
        ) from None


def tinker_cost_usd(tokens: dict[str, int], prices: dict[str, float]) -> float:
    """Price a {kind: token_count} mapping, e.g. {"sample": 1200}."""
    return sum(prices[kind] * count for kind, count in tokens.items()) / 1e6


def sample_tokens(
    prompt_length: int,
    num_samples: int,
    cache_hit_tokens: int,
    sampled_lengths: list[int],
) -> dict[str, int]:
    """Billed token kinds for one sample request.

    Tinker bills the shared prompt once at prefill rates, minus its reported
    prefix-cache hits; the other num_samples - 1 copies are cache hits. For
    example, a 100-token prompt with n=4 and no hits bills 100 prefill and 300
    cached_prefill tokens.
    """
    hits = min(cache_hit_tokens, prompt_length)
    return {
        "prefill": prompt_length - hits,
        "cached_prefill": hits + prompt_length * (num_samples - 1),
        "sample": sum(sampled_lengths),
    }


def max_concurrent_samples() -> int:
    """In-flight Tinker sample requests per process (TINKER_MAX_CONCURRENT_SAMPLES)."""
    value = int(
        os.environ.get("TINKER_MAX_CONCURRENT_SAMPLES", DEFAULT_MAX_CONCURRENT_SAMPLES)
    )
    if value < 1:
        raise ValueError("TINKER_MAX_CONCURRENT_SAMPLES must be positive")
    return value


class TinkerUsageLedger:
    """Share Tinker spend across processes; refuse calls that would exceed the limit.

    One ledger covers one base model and prices it with the built-in rate card,
    whatever prices a caller reports. Calls made through `reserve` in one
    process never jointly overshoot the limit; calls in other processes can
    overshoot it by at most their own in-flight cost. Needs POSIX file locks.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        limit: float,
        *,
        model: str,
        name: str = "tinker",
    ) -> None:
        """Bind a ledger file to a positive dollar limit for one base model."""
        if isinstance(limit, bool):
            raise ValueError("Tinker dollar limit must be positive and finite")
        limit = float(limit)
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("Tinker dollar limit must be positive and finite")
        if not path:
            raise ValueError("Tinker usage file is required")
        self.limit, self.model, self.name = limit, model, name
        self.prices = tinker_prices(model)
        self.path = Path(path).absolute()

    @classmethod
    def env_config(cls) -> tuple[str, float] | None:
        """Read (path, limit) from FORGER_TINKER_USAGE_FILE / FORGER_TINKER_USD_LIMIT.

        Returns None when neither is set; raises when only one is set.
        """
        path = os.environ.get("FORGER_TINKER_USAGE_FILE")
        limit = os.environ.get("FORGER_TINKER_USD_LIMIT")
        if path is None and limit is None:
            return None
        if not path or not limit:
            raise ValueError(
                "Set both FORGER_TINKER_USAGE_FILE and FORGER_TINKER_USD_LIMIT "
                "to enable the shared Tinker budget."
            )
        return path, float(limit)

    def cost(self, tokens: dict[str, int]) -> float:
        """Price billed tokens at this ledger's rate card."""
        return tinker_cost_usd(tokens, self.prices)

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

    def _read(self) -> dict[str, Any]:
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
                "tokens": {kind: 0 for kind in self.prices},
                "sessions": [],
            }
        with os.fdopen(descriptor) as stream:
            state = json.load(stream)
        if (state.get("version"), state.get("limit_usd"), state.get("model")) != (
            1,
            self.limit,
            self.model,
        ):
            raise ValueError(
                f"Tinker ledger {self.path} is for {state.get('model')} with limit "
                f"${state.get('limit_usd')}, not {self.model} with ${self.limit}"
            )
        cost = state.get("cost_usd")
        if (
            isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(cost)
            or cost < 0
            or not isinstance(state.get("tokens"), dict)
        ):
            raise ValueError("Invalid Tinker ledger")
        return state

    def _write(self, state: dict[str, Any]) -> None:
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

    def _snapshot(self, state: dict[str, Any]) -> dict[str, Any]:
        return {**state, "remaining_usd": max(0.0, self.limit - state["cost_usd"])}

    def snapshot(self) -> dict[str, Any]:
        """Read spend, creating an empty ledger on first use."""
        with self._locked():
            state = self._read()
            if not self.path.exists():
                self._write(state)
            return self._snapshot(state)

    def check(self, estimate_usd: float = 0.0) -> dict[str, Any]:
        """Raise unless spend plus this call's estimated cost stays within the limit."""
        state = self.snapshot()
        if state["cost_usd"] + estimate_usd > self.limit:
            name = state.get("budget", self.name)
            raise TinkerBudgetExceeded(
                f"Tinker {name} budget: ${state['cost_usd']:.2f} spent of "
                f"${self.limit:.2f}; next call needs up to ${estimate_usd:.2f}"
            )
        return state

    @contextlib.contextmanager
    def reserve(self, estimate_usd: float):
        """Hold a call's worst-case cost against the limit while it runs.

        The check counts this process's other reservations on the same file, so
        e.g. 600 concurrent requests with $1 left cannot all start. When the
        call does not fit but others are in flight, it waits for them to finish
        and be recorded; when none are, it raises `TinkerBudgetExceeded`. Record
        the call's tokens before leaving the block.
        """
        with _RESERVED_CHANGED:
            while True:
                count, reserved = _RESERVED.get(self.path, (0, 0.0))
                try:
                    self.check(reserved + estimate_usd)
                except TinkerBudgetExceeded:
                    if not count:
                        raise
                    _RESERVED_CHANGED.wait()
                    continue
                _RESERVED[self.path] = (count + 1, reserved + estimate_usd)
                break
        try:
            yield
        finally:
            with _RESERVED_CHANGED:
                count, reserved = _RESERVED.pop(self.path)
                if count > 1:
                    _RESERVED[self.path] = (count - 1, reserved - estimate_usd)
                _RESERVED_CHANGED.notify_all()

    def record(
        self, tokens: dict[str, int], session_id: str | None = None
    ) -> dict[str, Any]:
        """Charge one call's billed tokens, e.g. sample_tokens(...)."""
        cost = self.cost(tokens)
        with self._locked():
            state = self._read()
            for kind, count in tokens.items():
                state["tokens"][kind] = state["tokens"].get(kind, 0) + int(count)
            state["cost_usd"] += cost
            state["calls"] += 1
            if session_id and session_id not in state["sessions"]:
                state["sessions"].append(session_id)
            self._write(state)
            return self._snapshot(state)
