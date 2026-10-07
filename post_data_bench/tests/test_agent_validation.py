import ast
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import base64
import zlib
from coding.split_contract import write_manifest
import unittest

from coding.rl.agent_validation import EVALUATOR_FILES, VALIDATION_FIELDS, copy_validation_evaluator

ROOT = Path(__file__).resolve().parents[1]


class AgentValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / 'source'
        self.dest = Path(self.tmp.name) / 'agent'
        self.source.mkdir()
        self.rows = []
        for index in (3, 1, 3):
            row = {name: '' for name in VALIDATION_FIELDS}
            row.update(private_test_case_idx=index, gold_response=base64.b64encode(zlib.compress(b'gold')).decode(), question_id=f'public-{len(self.rows)}',
                       is_perturbed=True, internal_generation_notes='not-public')
            self.rows.append(row)
        (self.source / 'validation_tasks.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in self.rows))
        (self.source / 'private_test_cases.txt').write_text('secret-0\nvalidation-1\nsecret-2\nvalidation-3\nsecret-4\n')
        (self.source / '.hidden').mkdir()
        (self.source / '.hidden/test_tasks.jsonl').write_text(json.dumps({'private_test_case_idx': 99}) + '\n')
        for name in EVALUATOR_FILES:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# evaluator source\n')

        self.manifest()

    def manifest(self):
        write_manifest(self.source, [self.source / 'validation_tasks.jsonl', self.source / '.hidden/test_tasks.jsonl', self.source / 'private_test_cases.txt'], self.rows, [{'private_test_case_idx':99}], seed=1, inputs={})

    def test_validation_only_export_and_remapping(self):
        copy_validation_evaluator(self.source, self.dest)
        rows = [json.loads(s) for s in (self.dest / 'validation_tasks.jsonl').read_text().splitlines()]
        blobs = (self.dest / 'private_test_cases.txt').read_text().splitlines()
        self.assertEqual(blobs, ['validation-1', 'validation-3'])
        self.assertEqual([r['private_test_case_idx'] for r in rows], [1, 0, 1])
        for original, exported in zip(self.rows, rows):
            self.assertEqual(set(exported), set(VALIDATION_FIELDS))
            self.assertEqual(blobs[exported['private_test_case_idx']], f"validation-{original['private_test_case_idx']}")
            self.assertEqual(exported['gold_response'], original['gold_response'])
        files = {str(p.relative_to(self.dest)) for p in self.dest.rglob('*') if p.is_file()}
        self.assertEqual(files, set(EVALUATOR_FILES) | {'validation_tasks.jsonl', 'private_test_cases.txt', 'validation_tasks_de_obfuscated.jsonl', 'validation_manifest.json'})
        self.assertFalse((self.dest / '.hidden').exists())

    def test_missing_blob_fails_before_export(self):
        (self.source / 'private_test_cases.txt').write_text('secret-0\nvalidation-1\n')
        self.manifest()
        with self.assertRaisesRegex(ValueError, 'missing private-test'):
            copy_validation_evaluator(self.source, self.dest)
        self.assertFalse(self.dest.exists())

    def test_launcher_exports_evaluator_and_only_validation_history(self):
        # Exercise the launcher's actual copy functions without importing its
        # executable top level (which would start an OpenHands/GPU session).
        tree = ast.parse((ROOT / 'launch_agent.py').read_text())
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name in {'_copy_into', '_task_public_copies'}]
        namespace = dict(os=os, shutil=shutil, args=SimpleNamespace(training_mode='sft-rl'),
                         TASK_PUBLIC_FILES=[], VAL_OUTPUT_FILES=('predictions.jsonl', 'metrics.json'))
        exec(compile(ast.Module(body=selected, type_ignores=[]), 'launch_agent.py', 'exec'), namespace)
        for relative in ('Qwen__Qwen3-4B/val/metrics.json', 'Qwen__Qwen3-4B/val/predictions.jsonl',
                         'Qwen__Qwen3-4B/test/metrics.json', 'Qwen__Qwen3-4B/val/shard_0/predictions.jsonl'):
            path = self.source / 'model_outputs' / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}')
        baseline_path = self.source/'model_outputs/Qwen__Qwen3-4B/val/metrics.json'
        from coding.split_contract import sha256
        from coding.eval_state import CONTRACT_VERSION
        baseline_path.write_text(json.dumps(dict(evaluation_contract=CONTRACT_VERSION,tasks_sha256=sha256(self.source/'validation_tasks.jsonl'))))
        namespace['_task_public_copies'](str(self.source), str(self.dest))
        self.assertTrue((self.dest / 'evaluate_model.sh').exists())
        self.assertTrue((self.dest / 'model_outputs/Qwen__Qwen3-4B/val/metrics.json').exists())
        self.assertFalse((self.dest / 'model_outputs/Qwen__Qwen3-4B/test').exists())
        self.assertFalse((self.dest / 'model_outputs/Qwen__Qwen3-4B/val/shard_0').exists())


if __name__ == '__main__':
    unittest.main()
