import unittest

class RayCompletionTests(unittest.TestCase):
    def test_only_succeeded_is_success(self):
        from coding.rl.ray_job_status import require_success
        require_success('SUCCEEDED')
        for status in ('STOPPED','FAILED','RUNNING','PENDING'):
            with self.assertRaises(SystemExit): require_success(status)
