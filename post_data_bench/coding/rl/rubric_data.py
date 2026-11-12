"""Validate prompt/rubric rows and compile the RL (and Tinker SFT) training data."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
from functools import lru_cache
from pathlib import Path

# Rows per training run (and in the final dataset); experiments are unlimited.
MAX_ROWS = 2000
# Tasks with a smaller cap, by the launcher's public task name ($PDB_TASK).
TASK_MAX_ROWS = {"algorithmic": 800}


# Row length cap in tokens: the prompt for RL, prompt + target for Tinker SFT.
# Algorithmic answers run ~15K tokens at the median, so SFT targets there could
# use Qwen3's whole 40,960 context.
MAX_SFT_LENGTH = 18800
TASK_MAX_SFT_LENGTH = {"algorithmic": 40960}


def max_sft_length(task=None):
    """The SFT row length cap for `task` (default: $PDB_TASK)."""
    task = task if task is not None else os.environ.get("PDB_TASK", "")
    return TASK_MAX_SFT_LENGTH.get(str(task).strip().lower(), MAX_SFT_LENGTH)


def max_rows(task=None):
    """The row cap for `task` (default: $PDB_TASK), e.g. 800 for algorithmic."""
    task = task if task is not None else os.environ.get("PDB_TASK", "")
    return TASK_MAX_ROWS.get(str(task).strip().lower(), MAX_ROWS)


def score(value):
    """Validate and normalize a finite numeric criterion score in [0, 1]."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(
            f"criterion score must be a finite number in [0, 1], got {value!r}"
        )
    return float(value)


@lru_cache(maxsize=128)
def validate_code(source):
    """Check the declared executor interface without importing or running code."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"invalid Python syntax: {exc.msg}") from exc
    entries = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "grade"
    ]
    if len(entries) != 1 or not isinstance(entries[0], ast.FunctionDef):
        raise ValueError(
            "code must define one top-level synchronous function named grade"
        )
    entry = entries[0]
    args = entry.args
    if (
        len(args.posonlyargs) + len(args.args) != 3
        or args.vararg
        or args.kwarg
        or args.kwonlyargs
        or entry.decorator_list
    ):
        raise ValueError(
            (
                "grade must have exactly three positional parameters and "
                "no decorators: grade(prompt, response, ground_truth)"
            )
        )
    # Parameter names and type annotations are optional. Reject obvious module
    # rebinding, which would replace the function selected by the executor.
    for node in tree.body:
        if node is entry:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            rebound = node.name == "grade"
        else:
            rebound = any(
                (
                    isinstance(item, ast.Name)
                    and item.id == "grade"
                    and isinstance(item.ctx, (ast.Store, ast.Del))
                )
                or (
                    isinstance(
                        item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    )
                    and item.name == "grade"
                )
                or (isinstance(item, ast.ExceptHandler) and item.name == "grade")
                or (
                    isinstance(item, ast.alias)
                    and (item.asname or item.name.split(".")[0]) == "grade"
                )
                for item in ast.walk(node)
            )
        if rebound:
            raise ValueError("code must not replace or delete the grade function")
    from RestrictedPython import compile_restricted

    try:
        compile_restricted(source, filename="<rubric>", mode="exec")
    except SyntaxError as exc:
        raise ValueError(
            f"code does not compile under RestrictedPython: {exc}"
        ) from exc


# Optional keys of a "prompt" criterion: answer_only strips thinking from the
# response and ground truth; judge_sees picks which of JUDGE_FIELDS the judge gets
# (default all, e.g. ["response"] for "mentions anticoagulation"); instructions
# are added to the judge's system message for this criterion only.
PROMPT_OPTIONS = ("answer_only", "judge_sees", "instructions")
# Tasks whose prompt criteria default to answer_only (by the launcher's public task
# name, $PDB_TASK): clinical answers are graded on their final text only.
ANSWER_ONLY_TASKS = ("clinical",)


def answer_only_default(task=None):
    """Whether a prompt criterion without "answer_only" grades final answers only."""
    task = task if task is not None else os.environ.get("PDB_TASK", "")
    return str(task).strip().lower() in ANSWER_ONLY_TASKS


JUDGE_FIELDS = ("prompt", "response", "ground_truth")
MAX_INSTRUCTIONS = 4000


def validate_rubric(rubric):
    """Validate criterion variants, code signatures and positive total weight."""
    if not isinstance(rubric, list) or not rubric:
        raise ValueError("rubric must be a nonempty list of [criterion, weight] pairs")
    total = 0.0
    for number, pair in enumerate(rubric, 1):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("each rubric entry must be [criterion, weight]")
        criterion, weight = pair
        if isinstance(criterion, dict) and "code" in criterion:
            extra = set(criterion) & set(PROMPT_OPTIONS)
            if extra:
                raise ValueError(
                    f'"{sorted(extra)[0]}" applies only to "prompt" criteria'
                )
        if not isinstance(criterion, dict) or not (
            set(criterion) == {"code"}
            or ("prompt" in criterion and set(criterion) <= {"prompt", *PROMPT_OPTIONS})
        ):
            raise ValueError('criterion must contain exactly one of "code" or "prompt"')
        source = criterion.get("code", criterion.get("prompt"))
        if not isinstance(source, str) or not source.strip():
            raise ValueError("criterion source must be nonempty text")
        if not isinstance(criterion.get("answer_only", False), bool):
            raise ValueError('"answer_only" must be true or false')
        sees = criterion.get("judge_sees", JUDGE_FIELDS)
        if (
            not isinstance(sees, (list, tuple))
            or "response" not in sees
            or len(set(sees)) != len(sees)
            or not set(sees) <= set(JUDGE_FIELDS)
        ):
            raise ValueError(
                '"judge_sees" must list distinct fields from "prompt", "response" '
                'and "ground_truth", including "response"'
            )
        instructions = criterion.get("instructions", "")
        if not isinstance(instructions, str) or len(instructions) > MAX_INSTRUCTIONS:
            raise ValueError(
                f'"instructions" must be text of at most {MAX_INSTRUCTIONS} characters'
            )
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError("weights must be finite, nonnegative numbers")
        if "code" in criterion:
            try:
                validate_code(criterion["code"])
            except ValueError as exc:
                raise ValueError(f"criterion {number}: {exc}") from exc
        total += weight
    if not math.isfinite(total) or total <= 0:
        raise ValueError("rubric weights must have a finite, positive sum")
    return rubric


def validate_row(row):
    """Validate and normalize one row: a prompt, a rubric, an optional reference.

    "ground_truth" is optional free text that criteria receive as their third
    argument (e.g. a reference answer); it becomes "" when absent.
    """
    if not isinstance(row, dict) or not (
        {"prompt", "rubric"} <= set(row) <= {"prompt", "ground_truth", "rubric"}
    ):
        raise ValueError(
            'row must have "prompt" and "rubric", and may have "ground_truth"'
        )
    prompt = row["prompt"]
    if isinstance(prompt, str) and prompt.strip():
        prompt = [{"role": "user", "content": prompt}]
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("prompt must be nonempty text or chat messages")
    for message in prompt:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] not in {"system", "user", "assistant"}
            or not isinstance(message["content"], str)
            or not message["content"].strip()
        ):
            raise ValueError("invalid prompt message")
    if prompt[-1]["role"] != "user":
        raise ValueError("prompt must end in a user turn")
    target = row.get("ground_truth", "")
    if not isinstance(target, str):
        raise ValueError("ground_truth must be a string")
    return {
        "prompt": prompt,
        "ground_truth": target,
        "rubric": validate_rubric(row["rubric"]),
    }


def validate_sft_target(target):
    """Tinker SFT trains on ground_truth: <think>reasoning</think>, then the answer."""
    # Thinking is part of the SFT target; never silently create empty traces.
    if not target.startswith("<think>") or "</think>" not in target:
        raise ValueError(
            "SFT needs each ground_truth to be <think>reasoning</think> followed by "
            "the answer"
        )
    reasoning, answer = target[len("<think>") :].split("</think>", 1)
    if not reasoning.strip() or not answer.strip():
        raise ValueError("ground_truth must contain both reasoning and a final answer")


def read_rows(path):
    """Load validated triples with source line numbers in error messages."""
    rows = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(validate_row(json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    if not rows:
        raise ValueError(f"{path}: empty dataset")
    return rows


def prompt_id(row):
    """Hash normalized prompt messages to detect duplicates and split leakage."""
    return hashlib.sha256(
        json.dumps(row["prompt"], sort_keys=True).encode()
    ).hexdigest()


def validate_dataset(source):
    """Validate the submission contract without execution, API calls or outputs."""
    rows = read_rows(source)
    if len(rows) < 10:
        raise ValueError("submit at least 10 distinct prompts for the 90/10 split")
    limit = max_rows()
    if len(rows) > limit:
        raise ValueError(f"{len(rows)} rows; each dataset may have at most {limit}")
    if len({prompt_id(r) for r in rows}) != len(rows):
        raise ValueError(
            "duplicate prompts are not allowed (prevents train/eval leakage)"
        )
    return rows


def compile_data(source, out_dir, tokenizer=None, max_length=None, sft=False):
    """Fixed seed, 90/10 split. Fail on overlength rows rather than truncate them.

    RL checks each prompt's length; `sft=True` (Tinker SFT) checks prompt plus
    ground_truth, requires every ground_truth to be a thinking target, and also
    writes sft.jsonl.

    Rows that overlap the held-out test problems are removed first (see
    `decontaminate`); only their number is reported.
    """
    from .decontaminate import filter_rows  # noqa: ABS101

    if max_length is None:
        max_length = max_sft_length()
    rows, removed = filter_rows(validate_dataset(source))
    if removed:
        print(
            f"decontamination: removed {removed} row(s) that overlap the held-out "
            "test problems"
        )
    if len(rows) < 10:
        raise ValueError(
            f"{len(rows)} rows remain after decontamination; submit at least 10"
        )
    # Sort before shuffling so file reordering cannot alter the split.
    ordered = sorted(rows, key=prompt_id)
    random.Random(0).shuffle(ordered)
    n_eval = max(1, len(ordered) // 10)
    evaluation, train = ordered[:n_eval], ordered[n_eval:]
    for row in train + evaluation:
        if sft:
            try:
                validate_sft_target(row["ground_truth"])
            except ValueError as exc:
                raise ValueError(f"prompt {prompt_id(row)}: {exc}") from exc
        if tokenizer is not None:
            messages = row["prompt"] + (
                [{"role": "assistant", "content": row["ground_truth"]}] if sft else []
            )
            length = len(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=not sft,
                )
            )
            if length > max_length:
                raise ValueError(
                    f"prompt {prompt_id(row)}: {length} tokens exceeds {max_length}"
                )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)

    def write(name, rows):
        (out / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )

    write("submitted.jsonl", rows)
    if sft:
        write(
            "sft.jsonl",
            [
                {
                    "messages": r["prompt"]
                    + [{"role": "assistant", "content": r["ground_truth"]}]
                }
                for r in train
            ],
        )
    for filename, rows in [("train.jsonl", train), ("val.jsonl", evaluation)]:
        write(
            filename,
            [
                {
                    "prompt": r["prompt"],
                    "label": r["ground_truth"],
                    "metadata": {"rubric": r["rubric"], "original_prompt": r["prompt"]},
                }
                for r in rows
            ],
        )
    manifest = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.glob("*.jsonl"))
    }
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "sha256": manifest,
                "train_rows": len(train),
                "eval_rows": len(evaluation),
                "thinking": True,
                "split_seed": 0,
            },
            indent=2,
        )
        + "\n"
    )
    return out


def main():
    """Run the command-line utility described in the module documentation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--out-dir", required=True, help="New directory; refuses to overwrite"
    )
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    print(compile_data(args.data, args.out_dir, tokenizer, args.max_length))


if __name__ == "__main__":
    main()
