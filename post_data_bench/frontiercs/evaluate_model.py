#!/usr/bin/env python3
"""Evaluate C++ solutions to algorithmic problems through an OpenAI-compatible server.

Write resumable predictions.jsonl and metrics.json with headline ``score``: the mean
problem score (0-100), each problem judged on all its tests by judge.py, as the
benchmark scores it. Secondary metrics appear under context. A row's ``id`` names
its problem directory, problems/<id>/.

Live mode queries concurrently, judges each answer as it arrives, and checkpoints
every --checkpoint-every rows. --from-raw rejudges existing outputs by task ID
without inference. Shard launchers skip completed shards; live runs skip saved
predictions.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import checkers
import judge
import prompt_building as pb
from eval_state import atomic_json, read_predictions, stamp_metrics
from openai import AsyncOpenAI

DATA_DIR = Path(os.environ.get("PDB_BENCHMARK_DIR", Path(__file__).resolve().parent))


def parse_args() -> argparse.Namespace:
    """Parse CLI args shared by the live, --from-raw, and --merge-shards run modes."""
    p = argparse.ArgumentParser(
        description="Evaluate a model on C++ problems via a vLLM server."
    )
    p.add_argument("--validate-only", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("PDB_STUDENT_MODEL") or "Qwen/Qwen3-4B",
        help="Model id/path; must match the name the vLLM server serves.",
    )
    p.add_argument("--split", type=str, choices=["val", "test"], default=None)
    p.add_argument(
        "--tasks-file",
        type=str,
        default=None,
        help="Override the default validation_tasks.jsonl/test_tasks.jsonl path.",
    )
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on rows, applied before sharding.",
    )
    p.add_argument("--max-tokens", type=int, default=28672)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--max-connections", type=int, default=128)
    p.add_argument(
        "--judge-processes",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 4),
        help="Problems judged at once; each runs up to 4 tests in parallel.",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Append newly scored rows to log_dir/predictions.jsonl after this many "
        "complete, instead of only at the end. Rows already present there on "
        "startup are skipped (resumable/preemptible runs). Requires --log-dir.",
    )
    p.add_argument(
        "--enable-thinking", dest="enable_thinking", action="store_true", default=None
    )
    p.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    p.add_argument("--json-output-file", type=str, default=None)
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--merge-shards", type=str, nargs="+", default=None)
    p.add_argument(
        "--out-dir", type=str, default=None, help="Output dir for --merge-shards mode."
    )
    p.add_argument(
        "--from-raw",
        type=str,
        default=None,
        help="Path to an existing predictions.jsonl (id/output schema) to rejudge "
        "against --split's task ids, instead of running live inference.",
    )
    return p.parse_args()


def split_root(split: str) -> Path:
    """Where a split's problem directories and judge cache live."""
    return DATA_DIR if split == "val" else DATA_DIR / ".hidden"


def load_tasks(args: argparse.Namespace) -> list[dict]:
    """Load this run's task rows: --tasks-file, or the default file for --split."""
    tasks_file = args.tasks_file
    if tasks_file is None:
        from split_contract import verify_manifest

        verify_manifest(
            DATA_DIR,
            public=args.split == "val"
            and (DATA_DIR / "validation_manifest.json").exists(),
        )
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


def load_existing_predictions(log_dir: str | None) -> list[dict]:
    """Load saved predictions for row-level resume and complete metrics."""
    if not log_dir:
        return []
    return read_predictions(os.path.join(log_dir, "predictions.jsonl"))


def judge_code(split: str, problem_id: str, code: str) -> dict:
    """Judge one program on one problem (runs in a worker process)."""
    root = split_root(split)
    return judge.judge(
        root / "problems" / problem_id,
        code,
        testlib_dir=DATA_DIR / "testlib",
        cache_dir=root / ".judge_cache",
        workers=4,
    )


def _score_row(
    row: dict, *, output: str | None, code: str, result: dict | None
) -> dict:
    """One prediction row: the completion, its program, and the judge's verdict."""
    if not code or result is None:
        status, score, unbounded, passed, cases = "no_code", 0.0, 0.0, False, []
    else:
        status = result["status"]
        score, unbounded = result["score"], result["score_unbounded"]
        passed, cases = result["passed"], result["cases"]
    statuses: dict[str, int] = {}
    for c in cases:
        statuses[c["status"]] = statuses.get(c["status"], 0) + 1
    return {
        "id": row["id"],
        "type": row["type"],
        "question": row["task_text"],
        "output": output,
        "code": code,
        "status": status,
        "score": score,
        "score_unbounded": unbounded,
        "passed": passed,
        "n_cases": len(cases),
        "case_statuses": statuses,
        # One entry per test file, in judging order, e.g. {"test": "3.in",
        # "status": "Wrong Answer", "ratio": 0.42, "time": 0.31, "msg": "points
        # Ratio: 0.42 ..."}: msg is the start of the checker's or interactor's
        # message, or of the program's stderr when the run itself failed.
        "cases": [
            {
                "test": c.get("test"),
                "status": c["status"],
                "ratio": round(c["ratio"], 6),
                "time": c["time"],
                "msg": (c.get("msg") or "")[:200],
            }
            for c in cases
        ],
    }


async def generate_and_score_all(
    args: argparse.Namespace, rows: list[dict]
) -> list[dict]:
    """Generate, then judge each answer as it arrives, checkpointing for resume."""
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "tok")
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=3600)

    extra_body: dict = {"min_p": args.min_p, "top_k": args.top_k}
    if args.enable_thinking is not None:
        extra_body["chat_template_kwargs"] = {"enable_thinking": args.enable_thinking}

    sem = asyncio.Semaphore(args.max_connections)
    pool = ProcessPoolExecutor(args.judge_processes)
    loop = asyncio.get_running_loop()
    pred_path = (
        os.path.join(args.log_dir, "predictions.jsonl") if args.log_dir else None
    )
    pending: list[dict] = []
    output_rows: list[dict] = []

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

    async def one(row: dict):
        async with sem:
            messages = pb.build_candidate_prompt(row["task_text"])
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
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 3 or getattr(e, "status_code", None) in (
                        400,
                        401,
                        403,
                        404,
                    ):
                        print(
                            f"[warn] request for id={row['id']} failed: "
                            f"{type(e).__name__}"
                        )
                        raise RuntimeError(
                            "Evaluation request exhausted retries"
                        ) from e
                    await asyncio.sleep(2 * (attempt + 1))
        completion = resp.choices[0].message.content or ""
        code = checkers.extract_code(completion)
        result = (
            await loop.run_in_executor(pool, judge_code, args.split, row["id"], code)
            if code
            else None
        )
        out = _score_row(row, output=completion, code=code, result=result)
        out["finish_reason"] = resp.choices[0].finish_reason
        out["truncated"] = resp.choices[0].finish_reason == "length"
        output_rows.append(out)
        pending.append(out)
        flush()

    try:
        outcomes = await asyncio.gather(*(one(r) for r in rows), return_exceptions=True)
    finally:
        flush(force=True)
        pool.shutdown()

    failures = [item for item in outcomes if isinstance(item, BaseException)]
    if failures:
        raise RuntimeError(
            f"Evaluation incomplete: {len(failures)} request/grading failures "
            f"(first: {failures[0]!r}); successful rows saved"
        )
    return output_rows


def compute_metrics(graded_rows: list[dict]) -> dict:
    """Score graded rows; the mean problem score is the headline."""
    # Numeric ids in numeric order, e.g. "9" before "10"; "val_000" ids in order.
    graded_rows = sorted(graded_rows, key=lambda row: (len(row["id"]), row["id"]))
    scores = [float(r["score"]) for r in graded_rows]
    mean, stderr = checkers.mean_and_stderr(scores)
    by_type = {}
    for kind in ("default", "interactive"):
        values = [float(r["score"]) for r in graded_rows if r["type"] == kind]
        if values:
            by_type[kind] = {"n": len(values), "score": sum(values) / len(values)}
    unbounded = [float(r["score_unbounded"]) for r in graded_rows]
    return {
        "n_total": len(graded_rows),
        "n_truncated": sum(bool(r.get("truncated")) for r in graded_rows),
        "n_no_code": sum(r["status"] == "no_code" for r in graded_rows),
        "n_compile_error": sum(r["status"] == "compile_error" for r in graded_rows),
        # Headline: the benchmark compares models on the mean problem score.
        "score": mean,
        "score_stderr": stderr,
        "context": {
            "score_unbounded": sum(unbounded) / len(unbounded) if unbounded else 0.0,
            "n_zero": sum(s == 0 for s in scores),
            "n_at_least_60": sum(s >= 60 for s in scores),
            "n_all_tests_full": sum(bool(r["passed"]) for r in graded_rows),
            "by_type": by_type,
        },
    }


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
    headline = ("score", "score_stderr", "n_total", "n_no_code", "n_truncated")
    print(json.dumps({k: metrics[k] for k in headline}, indent=2))


def _finalize(output_rows: list[dict], args: argparse.Namespace, **extra) -> None:
    """Score output_rows, stamp on run metadata, and write predictions + metrics."""
    metrics = compute_metrics(output_rows)
    metrics["model"] = args.model_path
    metrics["split"] = args.split
    metrics.update(extra)
    _write_outputs(output_rows, metrics, args)


def run_merge(args: argparse.Namespace) -> None:
    """Recombine per-shard predictions.jsonl files into one merged output."""
    from eval_state import merge_rows

    all_rows = merge_rows(args, load_tasks, checkers.read_jsonl, "id")
    _finalize(all_rows, args, n_shards=len(args.merge_shards))


def run_from_raw(args: argparse.Namespace) -> None:
    """Rejudge saved outputs against split task IDs without querying the server."""
    rows = load_tasks(args)
    entry_by_id = {r["id"]: r for r in checkers.read_jsonl(args.from_raw)}
    jobs = []
    for row in rows:
        entry = entry_by_id.get(row["id"])
        completion = None if entry is None else entry.get("output", entry.get("raw"))
        jobs.append((row, completion, checkers.extract_code(completion)))
    with ProcessPoolExecutor(args.judge_processes) as pool:
        futures = [
            pool.submit(judge_code, args.split, row["id"], code) if code else None
            for row, _, code in jobs
        ]
        output_rows = [
            _score_row(
                row,
                output=completion,
                code=code,
                result=f.result() if f is not None else None,
            )
            for (row, completion, code), f in zip(jobs, futures)
        ]
    # Keep what only generation knew, e.g. that an answer hit max_tokens.
    for out in output_rows:
        entry = entry_by_id.get(out["id"]) or {}
        for key in ("finish_reason", "truncated"):
            if key in entry:
                out[key] = entry[key]
    n_missing = sum(1 for row in rows if row["id"] not in entry_by_id)
    if n_missing:
        print(
            f"[from-raw] WARNING: {n_missing}/{len(rows)} task id(s) missing "
            f"from {args.from_raw}"
        )
    _finalize(output_rows, args, source=args.from_raw)


def main() -> None:
    """Dispatch to the live, --from-raw, or --merge-shards run mode."""
    args = parse_args()
    if args.validate_only:
        return
    if args.merge_shards:
        run_merge(args)
        return
    if args.split is None:
        raise SystemExit("--split val|test is required (unless using --merge-shards)")
    if args.from_raw:
        run_from_raw(args)
        return

    rows = load_tasks(args)
    from eval_state import check_resume

    check_resume(args, rows)
    done_rows = load_existing_predictions(args.log_dir)
    expected_ids = {row["id"] for row in rows}
    if any(row["id"] not in expected_ids for row in done_rows):
        raise SystemExit(
            "Resume contains rows outside this shard; use a new output directory"
        )
    done_ids = {r["id"] for r in done_rows}
    if done_ids:
        n_before = len(rows)
        rows = [r for r in rows if r["id"] not in done_ids]
        print(
            f">>> resuming: {n_before - len(rows)} row(s) already scored, "
            f"{len(rows)} remaining"
        )
    print(
        f">>> {len(rows)} task(s) (split={args.split}, shard {args.shard_index}/"
        f"{args.num_shards}); querying server for {args.model_path}"
    )
    new_rows = asyncio.run(generate_and_score_all(args, rows)) if rows else []
    _finalize(done_rows + new_rows, args)


if __name__ == "__main__":
    main()
