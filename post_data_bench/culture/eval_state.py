"""Shared evaluation state contract, implemented alongside the coding evaluator."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coding.eval_state import (  # noqa: E402 - standalone script path setup
    atomic_json,
    check_resume,
    merge_rows,
    read_predictions,
    stamp_metrics,
)

__all__ = [
    ("atomic_json"),
    ("check_resume"),
    ("merge_rows"),
    ("read_predictions"),
    ("stamp_metrics"),
]
