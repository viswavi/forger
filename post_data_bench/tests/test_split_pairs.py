"""Exercise the real split builder with small original/perturbed pairs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SplitPairTests(unittest.TestCase):
    def test_pairs_and_stratified_membership(self):
        program = r'''
import json
import make_splits as m
import checkers
rows = [dict(question_id=str(i), question_title=f'Problem {i}', contest_id='fixture',
             contest_date='2024-01-01T00:00:00', platform='atcoder', difficulty='easy',
             starter_code='', metadata='{}', question_content='Compute.\nExample 1:\nInput: 1\nOutput: 2\nConstraints:\npositive input',
             public_test_cases=json.dumps([dict(input='1\n',output='2\n',testtype='stdin')]),
             private_test_cases='[]') for i in range(50)]
verdicts = {r['question_id']: dict(verified_solvable=True,gold_response='print(2)') for r in rows}
perturbed = {r['question_id']: dict(flipped_test_case_index=0,wrong_output='3',true_output_for_flipped_case='2\n') for r in rows}
val, test = m.build_split_files(rows, verdicts, perturbed)
assert val and test
for split in (val,test):
    sources = [r['private_test_case_idx'] for r in split]
    assert all(sources.count(source) == 2 for source in set(sources))
assert not {r['private_test_case_idx'] for r in val} & {r['private_test_case_idx'] for r in test}
assert all(set(r) == set(m.PUBLIC_FIELDS) for r in val + test)
assert len(val) == 10 and len(test) == 90
# Reordering inputs must not change source membership (indices follow input order).
val2, test2 = m.build_split_files(list(reversed(rows)), verdicts, dict(reversed(list(perturbed.items()))))
assert {r['question_id'] for r in val2} == {r['question_id'] for r in val}
assert {r['question_id'] for r in test2} == {r['question_id'] for r in test}
# A large fixture has exact 10% quotas for paired/unpaired and difficulty/status.
rows = [dict(rows[0], question_id=str(i), difficulty=('easy','medium','hard')[i % 3]) for i in range(600)]
verdicts = {r['question_id']: dict(verified_solvable=(int(r['question_id']) % 2 == 0), gold_response='print(2)') for r in rows}
perturbed = {r['question_id']: next(iter(perturbed.values())) for r in rows[:300]}
assignment = m._split_sources(rows, verdicts, perturbed)
from collections import Counter
counts = Counter((r['question_id'] in perturbed, r['difficulty'], verdicts[r['question_id']]['verified_solvable'], assignment[r['question_id']]) for r in rows)
for paired in (False, True):
    for difficulty in ('easy','medium','hard'):
        for solved in (False, True):
            assert counts[paired,difficulty,solved,'val'] == 5
            assert counts[paired,difficulty,solved,'test'] == 45
# Current benchmark size: nearest 10% row quota, keeping whole pairs.
rows = [dict(rows[0], question_id=str(i)) for i in range(1033)]
verdicts = {r['question_id']: dict(verified_solvable=True) for r in rows}
perturbed = {r['question_id']: {} for r in rows[:336]}
a = m._split_sources(rows, verdicts, perturbed)
assert sum(a[str(i)] == 'val' for i in range(336)) == 34
assert sum(a[str(i)] == 'val' for i in range(336,1033)) == 69
assert sum((2 if r['question_id'] in perturbed else 1) for r in rows if a[r['question_id']]=='val') == 137
# Small pools cannot be empty merely because rounding yields zero validation.
assert set(m._split_sources(rows[:2],verdicts,{}).values()) == {'val','test'}
assert set(m._split_sources(rows[:2],verdicts,{'0':{},'1':{}}).values()) == {'val','test'}
try:
    m._split_sources(rows[:1],verdicts,{'0':{}})
except ValueError:
    pass
else:
    raise AssertionError('One source cannot span two source-disjoint splits')
'''
        subprocess.run([sys.executable, '-c', program], cwd=ROOT/'coding', check=True)

    def test_existing_test_is_protected_before_reading_inputs(self):
        for task in ('coding', 'culture'):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/'.hidden').mkdir()
                path = root/'.hidden/test_tasks.jsonl'
                path.write_text('preserve this\n')
                result = subprocess.run([sys.executable, str(ROOT/task/'make_splits.py'), '--output-dir', str(root)], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('--force', result.stderr)
                self.assertEqual(path.read_text(), 'preserve this\n')


if __name__ == '__main__':
    unittest.main()
