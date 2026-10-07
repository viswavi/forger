"""Regression tests for evaluation recovery."""
import tempfile
import unittest
from pathlib import Path

from coding.eval_state import read_predictions


class PredictionRecoveryTests(unittest.TestCase):
    """Regression checks for prediction recovery."""

    def test_only_final_partial_record_is_discarded(self):
        """Verify only final partial record is discarded."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_bytes(b'{"id":1}\n{"id":')
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(read_predictions(path), [{"id": 1}])
            self.assertEqual(path.read_bytes(), b'{"id":1}\n')
            path.write_bytes(b'{"id":1}')
            self.assertEqual(read_predictions(path), [{"id": 1}])
            self.assertEqual(path.read_bytes(), b'{"id":1}\n')
            path.write_bytes(b'{"id":\n{"id":2}\n')
            with self.assertRaises(ValueError):
                read_predictions(path)
