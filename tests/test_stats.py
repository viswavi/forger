"""Intervals and tests behind inspect.runs() and compare_pipelines' verdicts."""

from __future__ import annotations

import pytest

from pyforger import _stats


def test_wilson_interval():
    """21 of 50 kept reads 0.42 but could be 0.29 to 0.56; 0 of 3 is not 0."""
    lo, hi = _stats.wilson(21, 50)
    assert (round(lo, 2), round(hi, 2)) == (0.29, 0.56)
    lo, hi = _stats.wilson(0, 3)
    assert lo == 0.0 and hi == pytest.approx(0.56, abs=0.01)
    assert _stats.wilson(0, 0) == (0.0, 1.0)


def test_mean_interval():
    """The mean ± 1.96·sd/√n; None under two values."""
    mean, half = _stats.mean_interval([1.0, 2.0, 3.0, 4.0])
    assert mean == 2.5 and half == pytest.approx(1.96 * (5 / 3 / 4) ** 0.5)
    assert _stats.mean_interval([3.0]) is None


def test_diff_interval_excludes_zero_only_for_a_clear_gap():
    """80 vs 109 of 120 is clear; 6 vs 4 of 10 is not."""
    lo, hi = _stats.diff_interval(80, 120, 109, 120)
    assert (round(lo, 2), round(hi, 2)) == (-0.34, -0.14)
    lo, hi = _stats.diff_interval(6, 10, 4, 10)
    assert lo < 0 < hi


def test_sign_test():
    """29 vs 2 is clearly lopsided; 6 vs 4 is a fair coin's split."""
    assert _stats.sign_test(29, 2) < 0.001
    assert _stats.sign_test(6, 4) == pytest.approx(0.754, abs=0.001)
    assert _stats.sign_test(0, 0) == 1.0
    assert _stats.fmt_p(0.0001) == "p < 0.001"
    assert _stats.fmt_p(0.03) == "p = 0.03"
    assert _stats.fmt_p(0.754) == "p = 0.75"
