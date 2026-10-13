"""Tests for the shared attempt outcome contract."""
from __future__ import annotations

import pytest

from pyforger.attempt_outcome import AttemptOutcome


def test_success_failure_and_roundtrip():
    """Model grades retain attribution across serialization."""
    for passed in (True, False):
        outcome = AttemptOutcome(passed=passed, attribution="model", reason="graded")
        assert outcome.counted and outcome.passed is passed
        assert AttemptOutcome.model_validate(outcome.model_dump(mode="json")) == outcome


@pytest.mark.parametrize(
    "passed,attribution",
    [(None, "model"), (False, "infrastructure"), (True, "task"), (None, "bogus")],
)
def test_invalid_contract_rejected(passed, attribution):
    """Ambiguous grades cannot enter success-rate measurements."""
    with pytest.raises(ValueError):
        AttemptOutcome(passed=passed, attribution=attribution, reason="fixture")
