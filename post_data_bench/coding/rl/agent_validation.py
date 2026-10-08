"""Build a self-contained validation evaluator for the agent's allowlisted workspace."""
from __future__ import annotations

import base64
import json
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
# Only fields consumed by the evaluator. Exclude perturbation bookkeeping.
VALIDATION_FIELDS = PUBLIC_FIELDS


def copy_validation_evaluator(source, destination, data_root=None):
    """Copy validation rows and their private-test blobs using evaluator filenames.

    Exclude secret test rows and unreferenced blobs from the agent workspace.
    """
    source, destination = Path(source), Path(destination)
    data_root = Path(data_root) if data_root else source
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
    for row in rows:
        row["private_test_case_idx"] = remap[row["private_test_case_idx"]]
    for name in EVALUATOR_FILES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    (destination / "validation_tasks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (destination / "private_test_cases.txt").write_text(
        "".join(blobs[index] + "\n" for index in indices)
    )

    decoded = [
        {
            **row,
            "gold_response": zlib.decompress(
                base64.b64decode(row["gold_response"])
            ).decode()
            if row["gold_response"] is not None
            else None,
        }
        for row in rows
    ]
    (destination / "validation_tasks_de_obfuscated.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in decoded)
    )
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
