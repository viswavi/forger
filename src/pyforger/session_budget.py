"""SDK-independent wall-clock and iteration accounting."""

import math
import time
from dataclasses import dataclass, field


@dataclass
class SessionBudget:
    """One iteration and monotonic wall-time budget for all repair turns."""

    max_iterations: int | None
    timeout_s: float
    iterations: int = field(default=0, init=False)
    _started: float = field(default_factory=time.monotonic, init=False)
    _exempted_s: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """Reject invalid budgets before executing a conversation."""
        if self.max_iterations is not None and (
            type(self.max_iterations) is not int or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer or None")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")

    def remaining_s(self) -> float:
        """Return remaining time or fail without beginning more work."""
        remaining = (
            self.timeout_s + self._exempted_s - (time.monotonic() - self._started)
        )
        if remaining <= 0:
            raise TimeoutError("Agent session wall-time budget exhausted")
        return remaining

    def exempt_elapsed_s(self, started: float) -> float:
        """Credit back wall time spent in infrastructure, not agent work.

        Setup, provisioning, and a stalled worker are charged to the caller
        that owns the deadline, not to the agent that never ran. Exempting the
        interval keeps a single infrastructure stall from consuming the budget
        the agent needed to do the task at all.
        """
        elapsed = max(0.0, time.monotonic() - started)
        self._exempted_s += elapsed
        return elapsed

    def consume_step(self) -> None:
        """Charge an SDK step across initial generation and every repair."""
        self.remaining_s()
        if self.max_iterations is not None and self.iterations >= self.max_iterations:
            raise RuntimeError("Agent session iteration budget exhausted")
        self.iterations += 1
