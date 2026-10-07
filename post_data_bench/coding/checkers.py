"""Shared LiveCodeBench grading and row conversion helpers."""
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

# Raw LCB fields: JSON-encoded tests/metadata and an ISO contest_date.
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
    """Build a CodeGenerationProblem, ignoring extra bookkeeping fields."""
    return CodeGenerationProblem(**{k: row[k] for k in PROBLEM_FIELDS})


def _serialize_tests(tests) -> str:
    return json.dumps(
        [
            {"input": t.input, "output": t.output, "testtype": t.testtype.value}
            for t in tests
        ]
    )


def problem_to_row(problem: CodeGenerationProblem) -> dict:
    """Restore the raw schema: enum values, JSON test lists, and ISO dates."""
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
    """Replace one public test output and normalize formatting.

    Used for fresh and frozen perturbations; make_splits harmonizes statements.
    """
    from harmonize import normalize_wrong_output

    tests = json.loads(row["public_test_cases"])
    true_output = tests[flipped_index]["output"]
    metadata = json.loads(row["metadata"])
    tests[flipped_index]["output"] = normalize_wrong_output(
        true_output, wrong_output, bool(metadata.get("func_name"))
    )
    perturbed_row = dict(row)
    perturbed_row["public_test_cases"] = json.dumps(tests)
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
    import os
    import shutil
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row) + "\n" for row in rows)
    if path.exists() and path.read_text() == content:
        return
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            shutil.copy2(path, path.with_name(path.name + ".bak"))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def normalize_private_test_cases(raw: str) -> str:
    """Normalize trusted LCB private tests to plain JSON before obfuscating.

    Some sources use pickle/zlib/base64. Unwrap that layer only at build time
    so readers need only decode_answer; never call this on untrusted data.
    """
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        return pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8"))))


def write_private_test_cases(path, blobs: list[str]) -> None:
    """Write one obfuscated blob per line; private_test_case_idx is its line index."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        for blob in blobs:
            f.write(blob + "\n")


def load_private_test_cases(path, indices: set[int] | None = None) -> dict[int, str]:
    """Map line indices to obfuscated blobs; decode entries with decode_answer.

    Pass indices to avoid loading unneeded lines from large private-test files.
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
    """Report whether all public and all private tests passed, separately.

    LCB runs public tests first and stops on failure; a missing result means
    that segment failed or was never reached.
    """
    sample = problem.get_evaluation_sample()
    results, _metadata = check_correctness(sample, code, timeout=timeout, debug=False)
    n_public = len(problem.public_test_cases)
    n_total = n_public + len(problem.private_test_cases)
    if not results:
        return {"passed_public": False, "passed_private": False}
    # Normalize bools, integer error codes, and NumPy wrappers as LCB does.
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
    """Return whether code passes every public and private test."""
    result = check_lcb_solution_split(problem, code, timeout=timeout)
    return result["passed_public"] and result["passed_private"]


def extract_code(completion: str) -> str:
    """Extract the last fenced code block with LCB’s generic extractor."""
    return _extract_code(completion, LMStyle.OpenAIChat)


def encode_answer(value: str) -> str:
    """Encode UTF-8 text as base64(zlib(value)) for storage.

    This limits casual answer leakage; it is not a security boundary.
    """
    return base64.b64encode(zlib.compress(value.encode("utf-8"))).decode("utf-8")


def decode_answer(blob: str) -> str:
    """Inverse of encode_answer."""
    return zlib.decompress(base64.b64decode(blob.encode("utf-8"))).decode("utf-8")
