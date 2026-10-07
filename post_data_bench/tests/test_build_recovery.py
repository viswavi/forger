import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from coding import split_contract as contract


class BuildRecoveryTests(unittest.TestCase):
    def populate(self, stage, value):
        validation = [{'private_test_case_idx':0,'question_id':'val'}]
        test = [{'private_test_case_idx':1,'question_id':'test'}]
        (stage/'validation_tasks.jsonl').write_text(json.dumps(validation[0])+'\n')
        (stage/'.hidden').mkdir()
        (stage/'.hidden/test_tasks.jsonl').write_text(json.dumps(test[0])+'\n')
        (stage/'private_test_cases.txt').write_text(value)
        contract.write_manifest(stage,[stage/'validation_tasks.jsonl',stage/'.hidden/test_tasks.jsonl',stage/'private_test_cases.txt'],validation,test,seed=1,inputs={})

    def test_failed_preparation_does_not_replace_published_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with contract.staged_build(root) as stage:
                self.populate(stage,'original')
            with self.assertRaises(RuntimeError):
                with contract.staged_build(root) as stage:
                    self.populate(stage,'replacement')
                    raise RuntimeError('interrupted compression')
            contract.verify_manifest(root)
            self.assertEqual((root/'private_test_cases.txt').read_text(),'original')

    def test_interrupted_publication_resumes_from_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(contract,'_finish_build',side_effect=RuntimeError('interrupted publication')):
                with self.assertRaises(RuntimeError):
                    with contract.staged_build(root) as stage:
                        self.populate(stage,'replacement')
            self.assertTrue((root/'.build_pending.json').exists())
            # Starting the next build recovers the complete pending version first.
            with self.assertRaises(RuntimeError):
                with contract.staged_build(root):
                    self.assertEqual((root/'private_test_cases.txt').read_text(),'replacement')
                    contract.verify_manifest(root)
                    raise RuntimeError('stop before doing more work')
            self.assertFalse((root/'.build_pending.json').exists())
