"""Request failures must never be persisted as incorrect model answers."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]

class RequestFailureTests(unittest.TestCase):
    def test_both_tasks_fail_without_persisting_request_errors(self):
        script = r'''
import asyncio
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
import tempfile
import evaluate_model as evaluation
class Unauthorized(Exception):
    status_code = 401
async def test():
    with tempfile.TemporaryDirectory() as temporary:
        args=NS(min_p=0,top_k=20,enable_thinking=True,max_connections=1,
                grade_workers=1,log_dir=temporary,checkpoint_every=1,model_path='fixture',
                max_tokens=10,temperature=1,top_p=1,grade_timeout=1)
        request=AsyncMock(side_effect=Unauthorized('fixture'))
        client=NS(chat=NS(completions=NS(create=request)))
        with patch.object(evaluation,'AsyncOpenAI',return_value=client):
            try:
                if hasattr(evaluation,'generate_and_grade_all'):
                    await evaluation.generate_and_grade_all(args,[{'question_id':'fixture'}],[[{'role':'user','content':'fixture'}]])
                else:
                    await evaluation.generate_and_score_all(args,[{'id':'fixture','task_text':'fixture'}])
            except RuntimeError:
                pass
            else:
                raise AssertionError('request error became a model score')
        assert request.await_count == 1
        assert not (Path(temporary)/'predictions.jsonl').exists()
asyncio.run(test())
'''
        for task in ('coding','culture'):
            with self.subTest(task=task):
                result=subprocess.run([sys.executable,'-c',script],cwd=ROOT/task,capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stderr)

if __name__ == '__main__':
    unittest.main()
