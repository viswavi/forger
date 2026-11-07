"""Small-sample statistics for reports: yield intervals, mean intervals, tests.

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


def diff_interval(
    kept_a: int, seen_a: int, kept_b: int, seen_b: int
) -> tuple[float, float]:
    """95% interval of yield A minus yield B (Newcombe's hybrid score method).

    e.g. 80 of 120 vs 109 of 120 → (-0.34, -0.14): B keeps clearly more.
    """
    pa = kept_a / seen_a if seen_a else 0.0
    pb = kept_b / seen_b if seen_b else 0.0
    la, ua = wilson(kept_a, seen_a)
    lb, ub = wilson(kept_b, seen_b)
    d = pa - pb
    return (
        d - math.sqrt((pa - la) ** 2 + (ub - pb) ** 2),
        d + math.sqrt((ua - pa) ** 2 + (pb - lb) ** 2),
    )


def sign_test(a: int, b: int) -> float:
    """Two-sided exact p-value that ``a`` vs ``b`` is a fair coin's split.

    For rows only one of two runs kept: 29 vs 2 → p < 0.001 (clearly
    different); 6 vs 4 → p = 0.75 (within noise).
    """
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def fmt_range(lo: float, hi: float) -> str:
    """``0.29–0.56``."""
    return f"{lo:.2f}–{hi:.2f}"


def fmt_p(p: float) -> str:
    """``p < 0.001``, ``p = 0.03``, ``p = 0.75``."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.2g}" if p < 0.1 else f"p = {p:.2f}"
