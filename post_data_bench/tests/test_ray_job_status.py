"""Regression tests for ray job status."""
import unittest


class RayCompletionTests(unittest.TestCase):
    """Regression checks for ray completion."""

    def test_only_succeeded_is_success(self):
        """Verify only succeeded is success."""
        from coding.rl.ray_job_status import require_success

        require_success("SUCCEEDED")
        for status in ("STOPPED", "FAILED", "RUNNING", "PENDING"):
            with self.assertRaises(SystemExit):
                require_success(status)
