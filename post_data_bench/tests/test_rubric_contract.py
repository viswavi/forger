"""Validate the executor contract without judging or executing agent code."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from coding.rl import rubric_data as data
from coding.rl import agent_training as training

ROOT = Path(__file__).resolve().parents[1]


def example(source, index=0):
    return {'prompt': f'Question {index}', 'ground_truth': '<think>Reasoning</think>Answer',
            'rubric': [[{'code': source}, 1], [{'prompt': 'Judge this criterion'}, 1]]}


class RubricContractTests(unittest.TestCase):
    def test_declared_interface_not_behavior(self):
        for source in (
            'raise RuntimeError("must not execute")\ndef grade(p, r, g):\n    raise RuntimeError("must not grade")',
            'def grade(p, r, g, /):\n    return 0.0',
            'def grade(p: list, r: str, g: str = "") -> float:\n    return 0.0',
            'def helper(value):\n    return value\ndef grade(p, r, g):\n    return helper(12)',
        ):
            with self.subTest(source=source):
                data.validate_row(example(source))

    def test_invalid_declarations_and_restricted_compilation(self):
        for source in (
            'def other(p, r, g):\n    return 0.0',
            'grade = lambda p, r, g: 0.0',
            'def grade(p, r):\n    return 0.0',
            'def grade(p, r, g, extra):\n    return 0.0',
            'def grade(p, r, *, g):\n    return 0.0',
            'def grade(*args):\n    return 0.0',
            'def grade(p, r, g, **kwargs):\n    return 0.0',
            'async def grade(p, r, g):\n    return 0.0',
            '@wrapper\ndef grade(p, r, g):\n    return 0.0',
            'def grade(p, r, g):\n    return 0.0\ndef grade(p, r, g):\n    return 1.0',
            'def grade(p, r, g):\n    return 0.0\ngrade = 1',
            'def grade(p, r, g):\n    return 0.0\nimport math as grade',
            'def grade(p, r, g):\n    return 0.0\nif True:\n    def grade(p):\n        return 0.0',
            'def grade(p, r, g):\n    return r.__class__',
            'def grade(',
        ):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, 'criterion 1:'):
                data.validate_row(example(source))

    def test_cli_validates_without_execution_or_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'submitted.jsonl'
            source = 'raise RuntimeError("must not execute")\ndef grade(p, r, g):\n    return 12'
            rows = [example(source, i) for i in range(10)]
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            command = [sys.executable, '-m', 'rl.agent_training', 'validate', '--data', str(path)]
            result = subprocess.run(command, cwd=ROOT/'coding', capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('10 rows', result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            rows[0]['rubric'][0][0]['code'] = 'def grade(p, r):\n    return 0.0'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            result = subprocess.run(command, cwd=ROOT/'coding', capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(f'{path}:1:', result.stderr)
            self.assertIn('criterion 1:', result.stderr)
            with self.assertRaises(ValueError):
                data.compile_data(path, Path(directory)/'compiled')
            self.assertFalse((Path(directory)/'compiled').exists())

    def test_host_preflight_rejects_invalid_compiled_rubric(self):
        with tempfile.TemporaryDirectory() as directory:
            row = {'metadata': {'rubric': [[{'code': 'def grade(p, r):\n    return 0.0'}, 1]]}}
            for name in ('train.jsonl', 'val.jsonl'):
                (Path(directory)/name).write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError, 'three positional'):
                training.reward_preflight(directory)


if __name__ == '__main__':
    unittest.main()
