"""Intervals behind inspect.runs()' yields and score means."""

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
