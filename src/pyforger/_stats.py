"""Small-sample statistics for reports: yield and mean intervals.

Reports quote 95% intervals so that a yield from 50 rows reads as the range it
is: ``0.42 (0.29–0.56)``. No imports from other pyforger modules.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Z95 = 1.96  # two-sided 95%


def wilson(kept: int, seen: int, z: float = Z95) -> tuple[float, float]:
    """95% interval of a yield of ``kept`` of ``seen`` rows (Wilson score).

    e.g. 21 of 50 → (0.29, 0.56); 0 of 3 → (0.00, 0.56).
    """
    if seen <= 0:
        return 0.0, 1.0
    p = min(max(kept / seen, 0.0), 1.0)
    z2 = z * z
    centre = (p + z2 / (2 * seen)) / (1 + z2 / seen)
    half = z * math.sqrt(p * (1 - p) / seen + z2 / (4 * seen * seen)) / (1 + z2 / seen)
    return max(0.0, centre - half), min(1.0, centre + half)


def mean_interval(values: Sequence[float]) -> tuple[float, float] | None:
    """``(mean, half-width)`` of the mean's 95% interval; None under 2 values.

    The normal approximation, mean ± 1.96·sd/√n: 3.2 ± 0.2 for 143 judge scores.
    """
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, Z95 * math.sqrt(var / n)


def fmt_range(lo: float, hi: float) -> str:
    """``0.29–0.56``."""
    return f"{lo:.2f}–{hi:.2f}"
