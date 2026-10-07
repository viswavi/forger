"""Shared evaluation state contract, implemented alongside the coding evaluator."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coding.eval_state import check_resume, merge_rows, read_predictions, atomic_json, stamp_metrics
