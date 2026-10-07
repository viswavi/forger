"""Non-unique-output sources must disappear before generation and splitting."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SourcePolicyTests(unittest.TestCase):
    def test_policy_generation_split_and_private_blob_slots(self):
        script = r'''
import json
from unittest.mock import patch
import checkers, gold_generation, make_splits as m
from source_policy import exclusion_reason, NON_UNIQUE_OUTPUTS
assert len(NON_UNIQUE_OUTPUTS) == 22
for qid in NON_UNIQUE_OUTPUTS:
    assert exclusion_reason({'question_id':qid})
assert exclusion_reason({'question_id':'future','question_content':'Print any valid arrangement.'})
assert exclusion_reason({'question_id':'future','question_content':'Return the indices in any order.'})
assert exclusion_reason({'question_id':'future','question_content':'Return the minimum cost of any valid division.'}) is None
assert exclusion_reason({'question_id':'future','question_content':'Operations may be performed in any order. Return the minimum cost.'}) is None
base = dict(question_title='fixture',contest_id='fixture',contest_date='2024-01-01T00:00:00',
            platform='atcoder',difficulty='hard',starter_code='',metadata='{}',
            question_content='Return 2.\nExample 1:\nInput: 1\nOutput: 2\nConstraints:\npositive',
            public_test_cases=json.dumps([dict(input='1\n', output='2\n',testtype='stdin')]),
            private_test_cases='[{"input":"1","output":"2","testtype":"stdin"}]')
rows = [dict(base, question_id='abc315_e')] + [dict(base, question_id=str(i),question_title=f'Problem {i}') for i in range(50)]
with patch.object(gold_generation, 'SOURCE_FILES', ['fixture']), patch.object(checkers,'read_jsonl',return_value=rows):
    assert [r['question_id'] for r in gold_generation.load_lcb_native_rows(limit=2)] == ['0','1']
verdicts={r['question_id']:dict(verified_solvable=True,gold_response='print(2)') for r in rows[1:]}
perturbations={r['question_id']:dict(flipped_test_case_index=0,wrong_output='3',true_output_for_flipped_case='2\n') for r in rows}
val,test=m.build_split_files(rows,verdicts,perturbations)
assert len(val)==10 and len(test)==90
assert all(r['private_test_case_idx']!=0 for r in val+test)
blobs=m.build_private_test_cases_blobs(rows)
assert checkers.decode_answer(blobs[0])=='[]'
assert json.loads(checkers.decode_answer(blobs[1]))[0]['output']=='2'
'''
        subprocess.run([sys.executable, '-c', script], cwd=ROOT/'coding', check=True)


if __name__ == '__main__':
    unittest.main()
