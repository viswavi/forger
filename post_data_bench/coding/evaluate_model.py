#!/usr/bin/env python3
"""Evaluate coding tasks through an existing OpenAI-compatible inference server.

Write resumable predictions.jsonl and blended accuracy in metrics.json.
Decoded gold_response alone determines impossibility: compare abstention for
impossible rows without executing code; otherwise run public/private tests.
Build prompts before decoding answers, and never include answers in prompts.
Private tests are indexed blobs in private_test_cases.txt.

--merge-shards recomputes combined metrics. Each shard skips saved predictions
and flushes new results every --checkpoint-every completions."""
from __future__ import annotations

from eval_state import read_predictions, atomic_json, stamp_metrics

import argparse
import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from answer_text import final_answer, is_abstention
import checkers
import prompt_building as pb
from openai import AsyncOpenAI
from tqdm import tqdm

DATA_DIR = Path(os.environ.get("PDB_BENCHMARK_DIR", Path(__file__).resolve().parent))

CATEGORIES_SOLVABLE = ("passed", "failed", "false_abstention")
CATEGORIES_IMPOSSIBLE = ("correct_abstention", "incorrect_on_impossible")


def parse_args() -> argparse.Namespace:
    """Parse CLI args for both the normal generate+grade mode and --merge-shards."""
    p = argparse.ArgumentParser(
        description="Evaluate a model on LiveCodeBench-plus-impossible via a vLLM "
        "server."
    )
    p.add_argument("--validate-only", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Model id/path; must match the name the vLLM server serves.",
    )
    p.add_argument("--split", type=str, choices=["val", "test"], default=None)
    p.add_argument(
        "--tasks-file",
        type=str,
        default=None,
        help="Override the default validation_tasks.jsonl/test_tasks.jsonl path.",
    )
    p.add_argument(
        "--private-test-cases-file",
        type=str,
        default=str(DATA_DIR / "private_test_cases.txt"),
        help="private_test_case_idx -> obfuscated private_test_cases lookup file "
        "(make_splits.py).",
    )
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on rows, applied before sharding.",
    )
    # Allow long Qwen3 traces to finish; match the launcher’s MAX_TOKENS.
    p.add_argument("--max-tokens", type=int, default=28672)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--max-connections", type=int, default=128)
    p.add_argument("--grade-workers", type=int, default=8)
    p.add_argument("--grade-timeout", type=int, default=6)
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Append newly graded rows to log_dir/predictions.jsonl after this many "
        "complete, instead of only at the end. Rows already present there on "
        "startup are skipped (resumable/preemptible runs). Requires --log-dir.",
    )
    p.add_argument(
        "--enable-thinking", dest="enable_thinking", action="store_true", default=None
    )
    p.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    p.add_argument("--json-output-file", type=str, default=None)
    p.add_argument("--log-dir", type=str, default=None)
    # Shard-merge mode: recombine already-graded predictions.jsonl files instead of
    # generating.
    p.add_argument("--merge-shards", type=str, nargs="+", default=None)
    p.add_argument(
        "--out-dir", type=str, default=None, help="Output dir for --merge-shards mode."
    )
    return p.parse_args()


def load_tasks(args: argparse.Namespace) -> list[dict]:
    """Load this shard's rows from validation_tasks.jsonl or test_tasks.jsonl."""
    tasks_file = args.tasks_file
    if tasks_file is None:
        from split_contract import verify_manifest
        verify_manifest(DATA_DIR, public=args.split == "val" and (DATA_DIR / "validation_manifest.json").exists())
        if args.split == "val":
            tasks_file = str(DATA_DIR / "validation_tasks.jsonl")
        else:
            tasks_file = str(DATA_DIR / ".hidden" / "test_tasks.jsonl")
    rows = checkers.read_jsonl(tasks_file)
    if not rows:
        raise SystemExit(f"Missing or empty task file: {tasks_file}")
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows[args.shard_index :: args.num_shards]


def is_impossible_row(row: dict) -> bool:
    """Return whether decoded gold_response is the abstention marker."""
    return row["gold_response"] is not None and (
        checkers.decode_answer(row["gold_response"]) == pb.ABSTENTION_MARKER
    )


def prepare_row(row: dict, private_test_cases: dict[int, str]) -> dict:
    """Decode private tests for possible rows; use an empty list for impossible rows.

    Build-time normalization removes LCB’s pickle layer. Leave gold_response
    encoded until grading, after the candidate prompt has been built."""
    row = dict(row)
    blob = private_test_cases.get(row["private_test_case_idx"])
    if blob is None and not is_impossible_row(row):
        raise ValueError("Missing private tests for possible row")
    row["private_test_cases"] = checkers.decode_answer(blob) if blob is not None else "[]"
    return row


_CORRECT_CATEGORIES = ("passed", "correct_abstention")


def grade_row(row: dict, completion: str, *, timeout: int, thinking: bool = True) -> dict:
    """Grade one completion; see the module docstring for the two grading paths."""
    completion = final_answer(completion, thinking=thinking)
    abstained = is_abstention(completion, pb.ABSTENTION_MARKER)

    if is_impossible_row(row):
        # Ground-truth comparison only -- no code is executed, no reconstructed
        # "true" problem, nothing that depends on how the row was made unsolvable.
        category = "correct_abstention" if abstained else "incorrect_on_impossible"
        return {
            "abstained": abstained,
            "code": "",
            "correctness_reason": category,
            "score": abstained,
            "passed_public_tests": None,
            "passed_private_tests": None,
        }

    code = checkers.extract_code(completion) if completion else ""
    has_code = bool(code.strip())
    if abstained:
        test_result = {"passed_public": None, "passed_private": None}
        category = "false_abstention"
    else:
        test_result = (
            checkers.check_lcb_solution_split(
                checkers.row_to_problem(row), code, timeout=timeout
            )
            if has_code
            else {"passed_public": False, "passed_private": False}
        )
        category = (
            "passed"
            if test_result["passed_public"] and test_result["passed_private"]
            else "failed"
        )

    return {
        "abstained": abstained,
        "code": code,
        "correctness_reason": category,
        "score": category in _CORRECT_CATEGORIES,
        "passed_public_tests": test_result["passed_public"],
        "passed_private_tests": test_result["passed_private"],
    }


def build_output_row(row: dict, messages: list[dict], completion: str, g: dict) -> dict:
    """Assemble one predictions.jsonl row from a graded row + its prompt/completion."""
    return {
        "question_id": row["question_id"],
        "question": messages[1]["content"],
        "output": completion,
        "code": g["code"],
        "score": g["score"],
        "abstained": g["abstained"],
        "correctness_reason": g["correctness_reason"],
        "passed_public_tests": g["passed_public_tests"],
        "passed_private_tests": g["passed_private_tests"],
        "difficulty": row["difficulty"],
        "contest_date": row["contest_date"],
    }


def load_existing_predictions(log_dir: str | None) -> list[dict]:
    """Load saved predictions to resume generation and retain complete metrics."""
    if not log_dir:
        return []
    return read_predictions(os.path.join(log_dir, "predictions.jsonl"))


async def generate_and_grade_all(
    args: argparse.Namespace, rows: list[dict], messages_list: list[list[dict]]
) -> list[dict]:
    """Generate and grade concurrently, checkpointing predictions for row-level resume."""
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "tok")
    # Concurrent long completions may exceed the SDK’s 600-second timeout.
    request_timeout = float(os.environ.get("VLLM_REQUEST_TIMEOUT", "1800"))
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=request_timeout)

    extra_body: dict = {"min_p": args.min_p, "top_k": args.top_k}
    if args.enable_thinking is not None:
        extra_body["chat_template_kwargs"] = {"enable_thinking": args.enable_thinking}

    sem = asyncio.Semaphore(args.max_connections)
    grade_executor = ThreadPoolExecutor(max_workers=args.grade_workers)
    loop = asyncio.get_running_loop()

    pred_path = (
        os.path.join(args.log_dir, "predictions.jsonl") if args.log_dir else None
    )
    pending: list[dict] = []
    output_rows: list[dict] = []
    pbar = tqdm(total=len(rows), desc="generating+grading", unit="task")

    def flush(force: bool = False) -> None:
        nonlocal pending
        if not pred_path or not pending:
            return
        if not force and len(pending) < args.checkpoint_every:
            return
        with open(pred_path, "a") as f:
            for r in pending:
                f.write(json.dumps(r) + "\n")
        pending = []

    async def one(row: dict, messages: list[dict]):
        async with sem:
            completion = ""
            try:
                for attempt in range(4):
                    try:
                        resp = await client.chat.completions.create(
                            model=args.model_path,
                            messages=messages,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            extra_body=extra_body,
                        )
                        completion = resp.choices[0].message.content or ""
                        break
                    except Exception as e:  # noqa: BLE001
                        if attempt == 3 or getattr(e, "status_code", None) in (400, 401, 403, 404):
                            qid = row.get("question_id")
                            print(f"[warn] request for {qid} failed after retries: {type(e).__name__}")
                            raise RuntimeError("Evaluation request exhausted retries") from e
                        else:
                            await asyncio.sleep(2 * (attempt + 1))

                try:
                    g = await loop.run_in_executor(
                        grade_executor,
                        lambda: grade_row(row, completion, timeout=args.grade_timeout, thinking=args.enable_thinking is not False),
                    )
                except Exception as e:  # noqa: BLE001
                    raise RuntimeError("Evaluation grader failed") from e
                out = build_output_row(row, messages, completion, g)
                out["finish_reason"] = resp.choices[0].finish_reason
                out["truncated"] = resp.choices[0].finish_reason == "length"
                output_rows.append(out)
                pending.append(out)
                flush()
            finally:
                pbar.update(1)

    try:
        outcomes = await asyncio.gather(
            *(one(r, m) for r, m in zip(rows, messages_list)), return_exceptions=True
        )
    finally:
        flush(force=True)
        pbar.close()
        grade_executor.shutdown(wait=True)

    failures = [item for item in outcomes if isinstance(item, BaseException)]
    if failures:
        raise RuntimeError(f"Evaluation incomplete: {len(failures)} request/grading failures; successful rows saved")
    return output_rows


def compute_metrics(graded_rows: list[dict]) -> dict:
    """Compute blended accuracy and category rates from correctness_reason."""
    n = len(graded_rows)
    n_impossible = sum(
        1 for r in graded_rows if r["correctness_reason"] in CATEGORIES_IMPOSSIBLE
    )
    n_possible = n - n_impossible
    n_correct = sum(1 for r in graded_rows if r["score"])
    n_correct_possible = sum(
        1
        for r in graded_rows
        if r["correctness_reason"] in CATEGORIES_SOLVABLE and r["score"]
    )
    n_correct_impossible = sum(
        1
        for r in graded_rows
        if r["correctness_reason"] in CATEGORIES_IMPOSSIBLE and r["score"]
    )

    category_counts = {
        cat: sum(1 for r in graded_rows if r["correctness_reason"] == cat)
        for cat in CATEGORIES_SOLVABLE + CATEGORIES_IMPOSSIBLE
    }

    return {
        "n_total": n,
        "n_possible": n_possible,
        "n_impossible": n_impossible,
        "accuracy": n_correct / n if n else 0.0,
        "pass@1_on_possible": (n_correct_possible / n_possible) if n_possible else None,
        "abstention_rate_on_impossible": (
            (n_correct_impossible / n_impossible) if n_impossible else None
        ),
        "category_counts": category_counts,
        "category_rates": {
            "possible": {
                cat: (category_counts[cat] / n_possible) if n_possible else None
                for cat in CATEGORIES_SOLVABLE
            },
            "impossible": {
                cat: (category_counts[cat] / n_impossible) if n_impossible else None
                for cat in CATEGORIES_IMPOSSIBLE
            },
        },
    }


_HEADLINE_KEYS = (
    "accuracy",
    "pass@1_on_possible",
    "abstention_rate_on_impossible",
    "n_total",
)


def _print_summary(metrics: dict) -> None:
    print(json.dumps({k: metrics[k] for k in _HEADLINE_KEYS}, indent=2))


def _write_outputs(
    output_rows: list[dict], metrics: dict, args: argparse.Namespace
) -> None:
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        pred_path = os.path.join(args.log_dir, "predictions.jsonl")
        checkers.write_jsonl(pred_path, output_rows)
        print(f">>> wrote per-instance predictions to {pred_path}")

    out_file = args.json_output_file
    if out_file is None and args.log_dir:
        out_file = os.path.join(args.log_dir, "metrics.json")
    if out_file:
        atomic_json(out_file, stamp_metrics(metrics, args, DATA_DIR))
        print(f">>> wrote metrics to {out_file}")

    _print_summary(metrics)


def run_merge(args: argparse.Namespace) -> None:
    """Recombine --merge-shards predictions.jsonl files into one file + metrics."""
    from eval_state import merge_rows
    all_rows = merge_rows(args, load_tasks, checkers.read_jsonl, 'question_id')
    metrics = compute_metrics(all_rows)
    metrics["model"] = args.model_path
    metrics["split"] = args.split
    metrics["n_shards"] = len(args.merge_shards)

    out_dir = args.out_dir or (args.log_dir or ".")
    os.makedirs(out_dir, exist_ok=True)
    merged_pred_path = os.path.join(out_dir, "predictions.jsonl")
    checkers.write_jsonl(merged_pred_path, all_rows)
    merged_metrics_path = os.path.join(out_dir, "metrics.json")
    atomic_json(merged_metrics_path, stamp_metrics(metrics, args, DATA_DIR))
    print(
        f"[merge] merged {len(all_rows)} row(s) from {len(args.merge_shards)} shard(s)"
    )
    print(f"[merge] wrote {merged_pred_path} and {merged_metrics_path}")
    _print_summary(metrics)


def main() -> None:
    """Generate + grade this shard's tasks, or recombine shards (--merge-shards)."""
    args = parse_args()
    if args.validate_only:
        return

    if args.merge_shards:
        run_merge(args)
        return

    if args.split is None:
        raise SystemExit("--split val|test is required (unless using --merge-shards)")

    rows = load_tasks(args)

    from eval_state import check_resume
    check_resume(args, rows)
    done_rows = load_existing_predictions(args.log_dir)
    expected_ids = {row['question_id'] for row in rows}
    if any(row['question_id'] not in expected_ids for row in done_rows):
        raise SystemExit("Resume contains rows outside this shard; use a new output directory")
    done_ids = {r["question_id"] for r in done_rows}
    if done_ids:
        n_before = len(rows)
        rows = [r for r in rows if r["question_id"] not in done_ids]
        print(
            f">>> resuming: {n_before - len(rows)} row(s) already graded in "
            f"{args.log_dir}/predictions.jsonl, {len(rows)} remaining"
        )

    needed_indices = {
        r["private_test_case_idx"] for r in rows if not is_impossible_row(r)
    }
    private_test_cases = checkers.load_private_test_cases(
        args.private_test_cases_file, indices=needed_indices
    )
    rows = [prepare_row(r, private_test_cases) for r in rows]
    print(
        f">>> {len(rows)} task(s) (split={args.split}, shard "
        f"{args.shard_index}/{args.num_shards}); querying server for "
        f"{args.model_path}"
    )

    if rows:
        messages_list = [
            pb.build_candidate_prompt(
                checkers.row_to_problem(row), include_abstention_instruction=True
            )
            for row in rows
        ]
        new_output_rows = asyncio.run(generate_and_grade_all(args, rows, messages_list))
    else:
        new_output_rows = []

    output_rows = done_rows + new_output_rows

    metrics = compute_metrics(output_rows)
    metrics["model"] = args.model_path
    metrics["split"] = args.split

    _write_outputs(output_rows, metrics, args)


if __name__ == "__main__":
    main()
