"""Build a self-contained validation evaluator for the agent's allowlisted workspace."""
from __future__ import annotations

import base64
import hashlib
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
# The C++ problem evaluator, flattened the same way; judge.py is rl/cpp_judge.py.
# Its own evaluate_model.sh sets the evaluation protocol (temperature 1.0, top-p
# 1.0, no top-k, 5 samples) before the shared launcher, whose defaults are coding's
# (temperature 0.6, top-p 0.95, top-k 20); coding's wrapper would apply those.
FRONTIERCS_EVALUATOR_FILES = {
    **{
        name: source
        for name, source in CULTURE_EVALUATOR_FILES.items()
        if not source.startswith("culture/")
    },
    "evaluate_model.sh": "frontiercs/evaluate_model.sh",
    "evaluate_model.py": "frontiercs/evaluate_model.py",
    "checkers.py": "frontiercs/checkers.py",
    "prompt_building.py": "frontiercs/prompt_building.py",
    "judge.py": "coding/rl/cpp_judge.py",
}
# Fields of a validation problem row that the agent sees; "id" is replaced and
# "problem_id" (the source's problem number) is dropped.
FRONTIERCS_FIELDS = (
    "type",
    "n_tests",
    "time_limit_s",
    "memory_mb",
    "statement",
    "task_text",
)
# The clinician-chat evaluator, flattened the same way; its judge.py grades rubric
# criteria (with the agent's judge quota there) and its evaluate_model.sh sets the
# evaluation protocol before running the shared launcher beside it.
HEALTHBENCH_EVALUATOR_FILES = {
    **{
        name: source
        for name, source in CULTURE_EVALUATOR_FILES.items()
        if not source.startswith("culture/")
    },
    "evaluate_model.sh": "healthbench/evaluate_model.sh",
    "evaluate_model.py": "healthbench/evaluate_model.py",
    "checkers.py": "healthbench/checkers.py",
    "prompt_building.py": "healthbench/prompt_building.py",
    "judge.py": "healthbench/judge.py",
    "grader_prompt.txt": "healthbench/grader_prompt.txt",
}
# Fields of a validation chat that the agent sees; "id" is replaced, and the
# dataset's labels (use case, type, difficulty, specialty) are left out.
HEALTHBENCH_FIELDS = ("messages", "rubric_items", "physician_response")
# The benchmark's name checked in copied chats; the general list would match
# ordinary words there, e.g. "the next frontier of treatment".
HEALTHBENCH_DATA_NAMES = re.compile(r"health\s*bench", re.I)
# Agents must never learn benchmark names (contamination risk). The sandbox copy
# renames the vendored grader package and scrubs names from code and comments,
# e.g. LiveCodeBench/lcb_runner/lm_styles.py -> grader_lib/grader_runner/lm_styles.py.
# Nor which sites the problems came from: an agent that knows could fetch
# near-copies of held-out problems. The Platform enum keeps distinct members,
# e.g. LEETCODE = "leetcode" -> source_a = "source_a".
BENCHMARK_NAMES = re.compile(
    r"livecodebench|lcb|impossible-?bench|normad|prbench|healthbench"
    r"|leetcode|atcoder|codeforces|frontier",
    re.I,
)
# Names checked in copied data files (problem files, testlib.h), which are copied
# byte for byte: validation statements may name the contest site they came from.
DATA_NAMES = re.compile(rb"frontier", re.I)
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


def public_row_ids(data_root):
    """Map host validation "id"s to the agent's, e.g. {1473: "val_017"}.

    Used for culture, frontiercs and healthbench rows, whose task key is "id".
    """
    with (Path(data_root) / "validation_tasks.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return {row["id"]: f"val_{number:03d}" for number, row in enumerate(rows)}


public_culture_ids = public_row_ids


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


def _read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def copy_frontiercs_evaluator(destination, data_root):
    """Copy the C++ problem evaluator, validation rows and problem files.

    Validation problems become val_000, val_001, ... in validation_tasks.jsonl and
    problems/<id>/ (config.yaml, testdata/, checker or interactor). Every copied
    problem file is checked against the build's problem_files.json, and the copy's
    validation_manifest.json pins the rows and files the evaluator reads.
    """
    destination, data_root = Path(destination), Path(data_root)
    verify_manifest(data_root)
    validation = _read_rows(data_root / "validation_tasks.jsonl")
    if not validation:
        raise ValueError("validation dataset is empty")
    assert_disjoint(validation, _read_rows(data_root / ".hidden/test_tasks.jsonl"))
    digests = json.loads((data_root / "problem_files.json").read_text())
    ids = public_row_ids(data_root)
    copies = {"testlib/testlib.h": "testlib/testlib.h"}
    for row in validation:
        prefix = f"problems/{row['id']}/"
        names = [name for name in digests if name.startswith(prefix)]
        if not any(name.startswith(prefix + "testdata/") for name in names):
            raise ValueError(f"validation problem {row['id']} has no test data")
        for name in names:
            if name != prefix + "statement.txt":
                copies[f"problems/{ids[row['id']]}/{name[len(prefix):]}"] = name
    root = Path(__file__).resolve().parents[2]
    for name, source in FRONTIERCS_EVALUATOR_FILES.items():
        _write_text(destination / name, (root / source).read_text(), root / source)
    for target, name in copies.items():
        data = (data_root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digests[name]:
            raise ValueError(f"{name} differs from problem_files.json")
        if DATA_NAMES.search(data):
            raise ValueError(f"benchmark name in {name}")
        (destination / target).parent.mkdir(parents=True, exist_ok=True)
        (destination / target).write_bytes(data)
    rows = [
        {"id": ids[row["id"]], **{name: row[name] for name in FRONTIERCS_FIELDS}}
        for row in validation
    ]
    (destination / "validation_tasks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    public_files = ["validation_tasks.jsonl", *copies]
    (destination / "validation_manifest.json").write_text(
        json.dumps(
            {"outputs": {name: sha256(destination / name) for name in public_files}}
        )
    )


def copy_healthbench_evaluator(destination, data_root):
    """Copy the clinician-chat evaluator and its validation chats with opaque ids.

    Rows become val_000, val_001, ... with only HEALTHBENCH_FIELDS; the copy's
    validation_manifest.json pins them. A benchmark name in any copied row fails
    the copy.
    """
    destination, data_root = Path(destination), Path(data_root)
    verify_manifest(data_root)
    validation = _read_rows(data_root / "validation_tasks.jsonl")
    if not validation:
        raise ValueError("validation dataset is empty")
    assert_disjoint(validation, _read_rows(data_root / ".hidden/test_tasks.jsonl"))
    ids = public_row_ids(data_root)
    rows = [
        {"id": ids[row["id"]], **{name: row[name] for name in HEALTHBENCH_FIELDS}}
        for row in validation
    ]
    text = "".join(json.dumps(row) + "\n" for row in rows)
    if HEALTHBENCH_DATA_NAMES.search(text):
        raise ValueError("benchmark name in the validation rows")
    root = Path(__file__).resolve().parents[2]
    for name, source in HEALTHBENCH_EVALUATOR_FILES.items():
        _write_text(destination / name, (root / source).read_text(), root / source)
    (destination / "validation_tasks.jsonl").write_text(text)
    (destination / "validation_manifest.json").write_text(
        json.dumps(
            {
                "outputs": {
                    "validation_tasks.jsonl": sha256(
                        destination / "validation_tasks.jsonl"
                    )
                }
            }
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
    if task == "frontiercs":
        return copy_frontiercs_evaluator(destination, data_root)
    if task == "healthbench":
        return copy_healthbench_evaluator(destination, data_root)
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
