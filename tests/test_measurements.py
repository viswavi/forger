"""Evaluation measurements are independent of data producers."""
from __future__ import annotations

import pytest

from pyforger.measurements import Measurement


def test_measurement_roundtrip_and_optional_diagnostics():
    """Plain grades and scores do not require models, agents, or files."""
    result = Measurement(score=0.75, reason="similarity", evidence={"rule": "v1"})
    assert Measurement.model_validate_json(result.model_dump_json()) == result
    assert result.passed is None
    assert Measurement(passed=False).score is None


@pytest.mark.parametrize(
    "field,value",
    [("passed", 1), ("score", float("inf")), ("evidence", {"x": float("nan")})],
)
def test_measurement_rejects_malformed_fields(field, value):
    """Pydantic handles field constraints without a custom validator."""
    with pytest.raises(ValueError):
        Measurement(**{field: value})
