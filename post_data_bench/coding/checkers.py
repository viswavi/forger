"""Shared grading and row<->problem conversion helpers.

Used by gold_generation.py, perturb.py, make_splits.py, and
evaluate_lcb_v6_plus_impossible.py.
"""
from __future__ import annotations

import base64
import json
import os
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np

_LCB_DIR = Path(__file__).resolve().parent / "LiveCodeBench"
if str(_LCB_DIR) not in sys.path:
    sys.path.insert(0, str(_LCB_DIR))

from lcb_runner.benchmarks.code_generation import CodeGenerationProblem  # noqa: E402
from lcb_runner.evaluation.compute_code_generation_metrics import (  # noqa: E402
    check_correctness,
)
from lcb_runner.lm_styles import LMStyle  # noqa: E402
from lcb_runner.utils.extraction_utils import (  # noqa: E402
    extract_code as _extract_code,
)

# The raw fields CodeGenerationProblem.__init__ expects, in the exact jsonl/HF schema
# (public_test_cases/private_test_cases/metadata as JSON-encoded strings, contest_date
# as an ISO string) -- i.e. what a line of test.jsonl..test6.jsonl looks like before
# __post_init__ decodes it.
PROBLEM_FIELDS = (
    "question_title",
    "question_content",
    "platform",
    "question_id",
    "contest_id",
    "contest_date",
    "starter_code",
    "difficulty",
    "public_test_cases",
    "private_test_cases",
    "metadata",
)


def row_to_problem(row: dict) -> CodeGenerationProblem:
    """Build a CodeGenerationProblem from a pipeline row.

    Extra bookkeeping keys on `row` (e.g. is_perturbed, gold_response) are ignored.
    """
    return CodeGenerationProblem(**{k: row[k] for k in PROBLEM_FIELDS})


def _serialize_tests(tests) -> str:
    return json.dumps(
        [
            {"input": t.input, "output": t.output, "testtype": t.testtype.value}
            for t in tests
        ]
    )


def problem_to_row(problem: CodeGenerationProblem) -> dict:
    """Re-serialize a (possibly hand-perturbed) CodeGenerationProblem to raw schema.

    Inverts CodeGenerationProblem.__post_init__: Enums back to their .value, the
    test-case lists back to JSON-encoded strings, contest_date back to an ISO string.
    """
    return {
        "question_title": problem.question_title,
        "question_content": problem.question_content,
        "platform": problem.platform.value,
        "question_id": problem.question_id,
        "contest_id": problem.contest_id,
        "contest_date": problem.contest_date.isoformat(),
        "starter_code": problem.starter_code,
        "difficulty": problem.difficulty.value,
        "public_test_cases": _serialize_tests(problem.public_test_cases),
        "private_test_cases": _serialize_tests(problem.private_test_cases),
        "metadata": json.dumps(problem.metadata),
    }


def build_perturbed_row(row: dict, flipped_index: int, wrong_output: str) -> dict:
    """Build a perturbed twin of `row`.

    One public_test_cases entry's output is replaced with `wrong_output`;
    question_content/starter_code/metadata/private_test_cases are untouched.

    Used both right after perturb.py gets a fresh wrong_output from the mutation LLM,
    and by make_splits.py when reconstructing a perturbation from the frozen
    perturbed_cases.jsonl (i.e. no LLM call at all, just this same deterministic
    construction from a previously-recorded wrong_output).
    """
    problem = row_to_problem(row)
    true_output = problem.public_test_cases[flipped_index].output
    problem.public_test_cases[flipped_index].output = wrong_output
    perturbed_row = problem_to_row(problem)
    perturbed_row["question_id"] = f"{row['question_id']}__perturbed"
    perturbed_row["source_question_id"] = row["question_id"]
    perturbed_row["is_perturbed"] = True
    perturbed_row["true_output_for_flipped_case"] = true_output
    perturbed_row["flipped_test_case_index"] = flipped_index
    return perturbed_row


def read_jsonl(path) -> list[dict]:
    """Read a jsonl file into a list of dicts. Returns [] if the file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows: list[dict]) -> None:
    """Write `rows` to `path`, one JSON object per line."""
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def normalize_private_test_cases(raw: str) -> str:
    """Collapse LiveCodeBench's own pre-obfuscation, if present, to plain JSON text.

    Some rows in the raw LCB dataset ship private_test_cases already
    pickle->zlib->base64 obfuscated by LCB itself (CodeGenerationProblem.__post_init__
    handles this transparently with a try/except json.loads-then-unpickle). If we
    then applied our own encode_answer on top unconditionally, decode_answer alone
    would leave that inner LCB layer in place, forcing every reader to keep a
    pickle.loads fallback around forever. Call this once, here, at build time
    (make_splits.py) on trusted just-downloaded raw data -- the only place in this
    codebase pickle.loads ever runs -- so every blob in private_test_cases.txt is
    single-layer (zlib->base64) and decode_answer is the only unwrap anyone else
    ever needs.
    """
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        return pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8"))))


def write_private_test_cases(path, blobs: list[str]) -> None:
    """Write obfuscated private_test_cases blobs, one per line, in index order.

    A validation/test-split row's `private_test_case_idx` is a 0-based line number
    into this file (see load_private_test_cases). Plain flat text, not jsonl: each
    blob is already base64 (checkers.encode_answer), whose alphabet never contains a
    newline or needs escaping, so one blob per line round-trips exactly without JSON
    framing overhead -- meaningful here since this file can run into the GBs.
    """
    with Path(path).open("w") as f:
        for blob in blobs:
            f.write(blob + "\n")


def load_private_test_cases(path, indices: set[int] | None = None) -> dict[int, str]:
    """Inverse of write_private_test_cases: index -> still-obfuscated blob.

    Decode an entry with decode_answer. Index into the returned dict with a row's
    `private_test_case_idx`. Pass `indices` to keep only those lines in memory --
    this file can run into the GBs, and a sharded caller (evaluate_lcb_v6_plus_
    impossible.py) only ever needs the indices its own rows reference, not the
    whole file.
    """
    with Path(path).open() as f:
        if indices is None:
            return {i: line.rstrip("\n") for i, line in enumerate(f)}
        return {i: line.rstrip("\n") for i, line in enumerate(f) if i in indices}


def require_litellm_api_key() -> str:
    """Fetch LITELLM_API_KEY from the environment, or exit with a clear message."""
    api_key = os.environ.get("LITELLM_API_KEY")
    if not api_key:
        raise SystemExit(
            "LITELLM_API_KEY is not set. Please provide it before running this script,"
            " e.g.:\n  export LITELLM_API_KEY=..."
        )
    return api_key


def require_litellm_base_url() -> str:
    """Fetch LITELLM_BASE_URL from the environment, or exit with a clear message."""
    base_url = os.environ.get("LITELLM_BASE_URL")
    if not base_url:
        raise SystemExit(
            "LITELLM_BASE_URL is not set. Please provide it before running this script,"
            " e.g.:\n  export LITELLM_BASE_URL=..."
        )
    return base_url


def check_lcb_solution_split(
    problem: CodeGenerationProblem, code: str, timeout: int = 6
) -> dict:
    """Run `code` against `problem`'s test cases, reporting public/private separately.

    Returns {"passed_public": bool, "passed_private": bool}: each True iff every test
    case in that segment passed. get_evaluation_sample() concatenates
    public_test_cases before private_test_cases, and LiveCodeBench's own grading
    (testing_util.grade_call_based/grade_stdio) runs tests in order and returns early
    on the first failure -- so a `results` list shorter than a segment always means
    that segment failed or was never reached, never "unknown".
    """
    sample = problem.get_evaluation_sample()
    results, _metadata = check_correctness(sample, code, timeout=timeout, debug=False)
    n_public = len(problem.public_test_cases)
    n_total = n_public + len(problem.private_test_cases)
    if not results:
        return {"passed_public": False, "passed_private": False}
    # Mirror the normalization compute_code_generation_metrics.py itself applies
    # before comparing against 0: results may contain plain bool, plain int error
    # codes (-1/-2/-4/-5), or numpy scalar/array wrappers depending on which grading
    # path (call-based vs stdio) ran.
    fixed = []
    for r in results:
        if isinstance(r, np.ndarray):
            r = r.item(0)
        if isinstance(r, np.bool_):
            r = bool(r)
        fixed.append(r)
    arr = np.array(fixed)
    passed_public = len(arr) >= n_public and bool(np.all(arr[:n_public] > 0))
    passed_private = len(arr) >= n_total and bool(np.all(arr[n_public:n_total] > 0))
    return {"passed_public": passed_public, "passed_private": passed_private}


def check_lcb_solution(
    problem: CodeGenerationProblem, code: str, timeout: int = 6
) -> bool:
    """Run `code` against every public+private test case in `problem`.

    True iff every test case passes. Works identically whether `problem` came
    straight from the dataset or is a hand-perturbed variant (perturb.py), since both
    are plain CodeGenerationProblem instances and this only depends on
    get_evaluation_sample().
    """
    result = check_lcb_solution_split(problem, code, timeout=timeout)
    return result["passed_public"] and result["passed_private"]


def extract_code(completion: str) -> str:
    """Pull a Python code block out of a raw model completion (last ``` pair).

    Reuses lcb_runner's own generic extractor (the branch used for any LMStyle other
    than the CodeLLaMa/base-model special cases) rather than reimplementing
    markdown-fence parsing.
    """
    return _extract_code(completion, LMStyle.OpenAIChat)


def encode_answer(value: str) -> str:
    """Obfuscate an answer (gold code, private test cases, or a poisoned test case's
    true output) at rest.

    base64(zlib(value)) -- light obfuscation against casual leakage/contamination
    when a downstream user inspects validation_tasks.jsonl/test_tasks.jsonl/
    private_test_cases.txt, not a security boundary. `value` is always already a
    plain str (code text, or a JSON-encoded test-case list per LiveCodeBench's own
    schema), so this just utf-8 encodes it directly -- no json.dumps/pickle wrapping,
    nothing to deserialize on the way back out.
    """
    return base64.b64encode(zlib.compress(value.encode("utf-8"))).decode("utf-8")


def decode_answer(blob: str) -> str:
    """Inverse of encode_answer."""
    return zlib.decompress(base64.b64decode(blob.encode("utf-8"))).decode("utf-8")
