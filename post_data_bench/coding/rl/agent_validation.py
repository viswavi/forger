"""Build a self-contained validation evaluator for the agent's allowlisted workspace."""
from __future__ import annotations

import base64
import json
import re
import shutil
import zlib
from pathlib import Path

# Supports both `rl` CLI imports and `coding.rl` library imports.
from ..split_contract import (  # noqa: ABS101
    PUBLIC_FIELDS,
    assert_disjoint,
    public_rows,
    sha256,
    verify_manifest,
)

# Explicit source-file allowlist: never copy the LCB checkout, datasets, or caches.
EVALUATOR_FILES = (
    "eval_launcher.sh",
    "eval_server.py",
    "answer_text.py",
    "eval_state.py",
    "split_contract.py",
    "harmonize.py",
    "evaluate_model.sh",
    "evaluate_model.py",
    "tinker_sampling.py",
    "checkers.py",
    "prompt_building.py",
    "LiveCodeBench/lcb_runner/lm_styles.py",
    "LiveCodeBench/lcb_runner/benchmarks/__init__.py",
    "LiveCodeBench/lcb_runner/benchmarks/code_generation.py",
    "LiveCodeBench/lcb_runner/benchmarks/code_execution.py",
    "LiveCodeBench/lcb_runner/benchmarks/test_output_prediction.py",
    "LiveCodeBench/lcb_runner/evaluation/__init__.py",
    "LiveCodeBench/lcb_runner/evaluation/compute_code_generation_metrics.py",
    "LiveCodeBench/lcb_runner/evaluation/compute_code_execution_metrics.py",
    "LiveCodeBench/lcb_runner/evaluation/compute_test_output_prediction_metrics.py",
    "LiveCodeBench/lcb_runner/evaluation/pass_k_utils.py",
    "LiveCodeBench/lcb_runner/evaluation/testing_util.py",
    "LiveCodeBench/lcb_runner/evaluation/utils_execute.py",
    "LiveCodeBench/lcb_runner/utils/extraction_utils.py",
)
# Culture's evaluator, flattened: the shared launcher and contract modules that the
# host imports from coding/ sit next to it, e.g. culture/evaluate_model.sh is
# coding's, which runs eval_launcher.sh from its own directory.
CULTURE_EVALUATOR_FILES = {
    "eval_launcher.sh": "coding/eval_launcher.sh",
    "eval_server.py": "coding/eval_server.py",
    "eval_state.py": "coding/eval_state.py",
    "split_contract.py": "coding/split_contract.py",
    "evaluate_model.sh": "coding/evaluate_model.sh",
    "evaluate_model.py": "culture/evaluate_model.py",
    "checkers.py": "culture/checkers.py",
    "prompt_building.py": "culture/prompt_building.py",
}
# Fields of a culture validation row that the agent sees; "id" is replaced.
CULTURE_FIELDS = (
    "country",
    "axis",
    "subaxis",
    "value",
    "rule_of_thumb",
    "situation",
    "task_text",
    "gold_label",
)
# Agents must never learn benchmark names (contamination risk). The sandbox copy
# renames the vendored grader package and scrubs names from code and comments,
# e.g. LiveCodeBench/lcb_runner/lm_styles.py -> grader_lib/grader_runner/lm_styles.py.
# Nor which sites the problems came from: an agent that knows could fetch
# near-copies of held-out problems. The Platform enum keeps distinct members,
# e.g. LEETCODE = "leetcode" -> source_a = "source_a".
BENCHMARK_NAMES = re.compile(
    r"livecodebench|lcb|impossible-?bench|normad|prbench|healthbench"
    r"|leetcode|atcoder|codeforces",
    re.I,
)
_SCRUB = (
    (re.compile(r"LiveCodeBench-plus-impossible"), "this benchmark"),
    (re.compile(r"livecodebench", re.I), "grader_lib"),
    (re.compile(r"lcb_runner"), "grader_runner"),
    (re.compile(r"lcb", re.I), "grader"),
    (re.compile(r"leetcode", re.I), "source_a"),
    (re.compile(r"codeforces", re.I), "source_b"),
    (re.compile(r"atcoder", re.I), "source_c"),
)


def scrub(text):
    """Replace benchmark names with neutral ones; used for paths and file contents."""
    for pattern, replacement in _SCRUB:
        text = pattern.sub(replacement, text)
    return text


def agent_path(name):
    """Where an EVALUATOR_FILES entry lands in the agent's copy."""
    return scrub(name)


def public_identity(number):
    """Opaque identity fields for the number-th validation row.

    The originals name the site, e.g. question_id "abc365_c" and contest_id
    "abc365"; grading reads none of them.
    """
    return {
        "question_id": f"val_{number:03d}",
        "question_title": f"Problem {number}",
        "contest_id": "",
        "platform": "source_a",
    }


def public_question_ids(data_root):
    """Map host validation ids to the agent's, e.g. {"abc365_c": "val_017"}."""
    with (Path(data_root) / "validation_tasks.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return {
        row["question_id"]: public_identity(number)["question_id"]
        for number, row in enumerate(rows)
    }


# Only fields consumed by the evaluator. Exclude perturbation bookkeeping.
VALIDATION_FIELDS = PUBLIC_FIELDS


def public_culture_ids(data_root):
    """Map host culture validation ids to the agent's, e.g. {1473: "val_017"}."""
    with (Path(data_root) / "validation_tasks.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return {row["id"]: f"val_{number:03d}" for number, row in enumerate(rows)}


def _write_text(path, text, mode_from=None):
    """Write a scrubbed file, refusing any benchmark name in its path or content."""
    text = scrub(text)
    if BENCHMARK_NAMES.search(text) or BENCHMARK_NAMES.search(str(path)):
        raise ValueError(f"benchmark name survives scrubbing in {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if mode_from is not None:
        shutil.copymode(mode_from, path)


def _write_public_validation(destination, rows, decode):
    """Write validation rows, their decoded copy, and the public manifest."""
    (destination / "validation_tasks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    decoded = [
        {
            **row,
            decode: zlib.decompress(base64.b64decode(row[decode])).decode()
            if row[decode] is not None
            else None,
        }
        for row in rows
    ]
    (destination / "validation_tasks_de_obfuscated.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in decoded)
    )


def copy_culture_evaluator(destination, data_root):
    """Copy the culture evaluator and its validation rows with opaque ids.

    `destination` gets a flat, self-contained copy that runs without ../coding.
    """
    destination, data_root = Path(destination), Path(data_root)
    verify_manifest(data_root)
    read = lambda path: [  # noqa: E731
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    validation = read(data_root / "validation_tasks.jsonl")
    if not validation:
        raise ValueError("validation dataset is empty")
    assert_disjoint(validation, read(data_root / ".hidden/test_tasks.jsonl"))
    ids = public_culture_ids(data_root)
    rows = [
        {"id": ids[row["id"]], **{name: row[name] for name in CULTURE_FIELDS}}
        for row in validation
    ]
    root = Path(__file__).resolve().parents[2]
    for name, source in CULTURE_EVALUATOR_FILES.items():
        _write_text(destination / name, (root / source).read_text(), root / source)
    _write_public_validation(destination, rows, "gold_label")
    public_files = ("validation_tasks.jsonl", "validation_tasks_de_obfuscated.jsonl")
    (destination / "validation_manifest.json").write_text(
        json.dumps(
            {"outputs": {name: sha256(destination / name) for name in public_files}}
        )
    )


def copy_validation_evaluator(source, destination, data_root=None, task="coding"):
    """Copy validation rows and their private-test blobs using evaluator filenames.

    Exclude secret test rows and unreferenced blobs from the agent workspace.
    """
    source, destination = Path(source), Path(destination)
    data_root = Path(data_root) if data_root else source
    if task == "culture":
        return copy_culture_evaluator(destination, data_root)
    verify_manifest(data_root)
    rows = []
    with (data_root / "validation_tasks.jsonl").open() as stream:
        for line in stream:
            if line.strip():
                raw = json.loads(line)
                row = public_rows([raw])[0]
                index = row["private_test_case_idx"]
                if type(index) is not int or index < 0:
                    raise ValueError(
                        "validation private-test indices must be nonnegative integers"
                    )
                rows.append(row)
    if not rows:
        raise ValueError("validation dataset is empty")
    test = [
        json.loads(line)
        for line in (data_root / ".hidden/test_tasks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert_disjoint(rows, test)
    indices = sorted({r["private_test_case_idx"] for r in rows})
    remap = {old: new for new, old in enumerate(indices)}
    blobs = {}
    with (data_root / "private_test_cases.txt").open() as stream:
        for index, line in enumerate(stream):
            if index in remap:
                blobs[index] = line.rstrip("\n")
            if index >= indices[-1]:
                break
    if set(blobs) != set(indices):
        raise ValueError("validation references missing private-test blobs")
    for number, row in enumerate(rows):
        row["private_test_case_idx"] = remap[row["private_test_case_idx"]]
        row.update(public_identity(number))
    for name in EVALUATOR_FILES:
        target = destination / agent_path(name)
        _write_text(target, (source / name).read_text(), source / name)
    (destination / "private_test_cases.txt").write_text(
        "".join(blobs[index] + "\n" for index in indices)
    )
    _write_public_validation(destination, rows, "gold_response")
    public_files = (
        "validation_tasks.jsonl",
        "validation_tasks_de_obfuscated.jsonl",
        "private_test_cases.txt",
    )
    (destination / "validation_manifest.json").write_text(
        json.dumps(
            {"outputs": {name: sha256(destination / name) for name in public_files}}
        )
    )
