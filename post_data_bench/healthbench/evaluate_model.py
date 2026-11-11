#!/usr/bin/env python3
"""Evaluate a model's answers to clinician chats with rubric grading.

Write resumable predictions.jsonl and metrics.json with headline ``score`` (0-100).
Each chat gets --n-samples answers (default 8). The grader sees only the final
answer (the text after the last </think>; an answer cut off inside its reasoning is
graded as empty) and checks each of the row's criteria separately (judge.py). An
answer's score is the points of its met criteria over the row's positive points,
minus 2.94e-5 per character beyond 2,000 (plus the same per character short of
it); a row scores the mean over its answers, and ``score`` is 100 times the mean
row score clipped to [0, 1]. ``context.score_unadjusted`` leaves out the length
adjustment.

Live mode samples from an OpenAI-compatible server and grades each row as its
answers arrive, checkpointing every --checkpoint-every rows. --from-raw grades
saved outputs by task id without inference. Shard launchers skip completed shards;
live runs skip saved predictions.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import checkers
import judge
import prompt_building as pb
from eval_state import atomic_json, read_predictions, stamp_metrics

DATA_DIR = Path(os.environ.get("PDB_BENCHMARK_DIR", Path(__file__).resolve().parent))
LABELS = ("use_case", "type", "difficulty")


def parse_args() -> argparse.Namespace:
    """Parse CLI args shared by the live, --from-raw, and --merge-shards run modes."""
    p = argparse.ArgumentParser(description="Evaluate a model on clinician chats.")
    p.add_argument("--validate-only", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("PDB_STUDENT_MODEL") or "Qwen/Qwen3-4B",
        help="Model id/path; must match the name the server serves.",
    )
    p.add_argument("--split", type=str, choices=["val", "test"], default=None)
    p.add_argument("--tasks-file", type=str, default=None)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument(
        "--limit", type=int, default=None, help="Cap on rows, before sharding."
    )
    p.add_argument("--max-tokens", type=int, default=28672)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1, help="-1: no top-k.")
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--max-connections", type=int, default=128)
    p.add_argument(
        "--n-samples", type=int, default=8, help="Answers sampled per chat (Avg@n)."
    )
    p.add_argument(
        "--judge",
        choices=("official", "agent"),
        default=None,
        help="official: the host grader (secret test); agent: the judge-quota grader "
        "(default where the official key is absent).",
    )
    p.add_argument(
        "--judge-concurrency", type=int, default=64, help="Grader calls in flight."
    )
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument(
        "--enable-thinking", dest="enable_thinking", action="store_true", default=None
    )
    p.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    p.add_argument(
        "--keep-special-tokens",
        action="store_true",
        help="Keep special tokens in the text, for models whose reasoning delimiters "
        "are special tokens (Gemma 4's <|channel>...<channel|>).",
    )
    p.add_argument("--answer-extraction", default=None, help=argparse.SUPPRESS)
    p.add_argument("--json-output-file", type=str, default=None)
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--merge-shards", type=str, nargs="+", default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument(
        "--from-raw",
        type=str,
        default=None,
        help="Saved outputs to grade against --split's task ids: JSONL rows with "
        '"id" and "samples" ([{"text" or "output", "finish_reason"}]).',
    )
    args = p.parse_args()
    if args.judge is None:
        args.judge = judge.default_judge()
    if args.split == "test" and args.judge != "official" and not args.validate_only:
        p.error("the secret test set is graded only by the official judge")
    return args


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
        tasks_file = str(
            DATA_DIR / "validation_tasks.jsonl"
            if args.split == "val"
            else DATA_DIR / ".hidden" / "test_tasks.jsonl"
        )
    rows = checkers.read_jsonl(tasks_file)
    if not rows:
        raise SystemExit(f"Missing or empty task file: {tasks_file}")
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows[args.shard_index :: args.num_shards]


async def grade_row(
    args: argparse.Namespace, row: dict, outputs: list[dict], sem
) -> dict:
    """Grade every sampled answer of one row.

    `outputs` are {"output", "finish_reason"} dicts, one per sampled answer.
    """

    async def one(sample: dict) -> dict:
        raw = sample.get("output")
        answer = checkers.final_answer(raw)
        graded = await judge.grade_answer(
            args.judge, row["messages"], answer, row["rubric_items"], sem
        )
        return {
            "output": raw,
            "answer": answer,
            "finish_reason": sample.get("finish_reason"),
            "truncated": sample.get("finish_reason") == "length",
            "no_answer": not answer,
            **graded,
        }

    samples = await asyncio.gather(*(one(s) for s in outputs))
    n = len(samples)
    return {
        "id": row["id"],
        **{name: row[name] for name in LABELS if name in row},
        "n_samples": n,
        "score": sum(s["score_length_adjusted"] for s in samples) / n,
        "score_unadjusted": sum(s["score"] for s in samples) / n,
        "samples": list(samples),
    }


async def generate_and_grade_all(
    args: argparse.Namespace, rows: list[dict]
) -> list[dict]:
    """Sample each row's answers, grade them, and checkpoint for resume."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("VLLM_API_KEY", "tok"),
        timeout=7200,
    )
    extra_body: dict = {"min_p": args.min_p, "top_k": args.top_k}
    if args.enable_thinking is not None:
        extra_body["chat_template_kwargs"] = {"enable_thinking": args.enable_thinking}
    if args.keep_special_tokens:
        extra_body["skip_special_tokens"] = False
    gen_sem = asyncio.Semaphore(args.max_connections)
    judge_sem = asyncio.Semaphore(args.judge_concurrency)
    pred_path = (
        os.path.join(args.log_dir, "predictions.jsonl") if args.log_dir else None
    )
    pending: list[dict] = []
    output_rows: list[dict] = []

    def flush(force: bool = False) -> None:
        nonlocal pending
        if pred_path and pending and (force or len(pending) >= args.checkpoint_every):
            with open(pred_path, "a") as f:
                for r in pending:
                    f.write(json.dumps(r) + "\n")
            pending = []

    async def one(row: dict) -> None:
        async with gen_sem:
            for attempt in range(4):
                try:
                    resp = await client.chat.completions.create(
                        model=args.model_path,
                        messages=pb.build_candidate_prompt(row["messages"]),
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        n=args.n_samples,
                        extra_body=extra_body,
                    )
                    if len(resp.choices) != args.n_samples:
                        raise RuntimeError(
                            f"server returned {len(resp.choices)} of "
                            f"{args.n_samples} samples"
                        )
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 3 or getattr(e, "status_code", None) in (
                        400,
                        401,
                        403,
                        404,
                    ):
                        print(f"[warn] request for id={row['id']} failed: {e!r}"[:300])
                        raise RuntimeError(
                            "Evaluation request exhausted retries"
                        ) from e
                    await asyncio.sleep(2 * (attempt + 1))
        outputs = [
            {"output": c.message.content, "finish_reason": c.finish_reason}
            for c in resp.choices
        ]
        out = await grade_row(args, row, outputs, judge_sem)
        output_rows.append(out)
        pending.append(out)
        flush()

    try:
        outcomes = await asyncio.gather(*(one(r) for r in rows), return_exceptions=True)
    finally:
        flush(force=True)
    failures = [o for o in outcomes if isinstance(o, BaseException)]
    if failures:
        raise RuntimeError(
            f"Evaluation incomplete: {len(failures)} request/grading failures "
            f"(first: {failures[0]!r}); successful rows saved"
        ) from failures[0]
    return output_rows


def compute_metrics(graded_rows: list[dict]) -> dict:
    """Headline: 100 x the mean length-adjusted row score, clipped to [0, 1]."""
    graded_rows = sorted(graded_rows, key=lambda row: row["id"])
    scores = [float(r["score"]) for r in graded_rows]
    unadjusted = [float(r["score_unadjusted"]) for r in graded_rows]
    mean, stderr = checkers.mean_and_stderr(scores)
    samples = [s for r in graded_rows for s in r["samples"]]
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for s in samples:
        for c in s["criteria"]:
            for key in usage:
                usage[key] += c["usage"][key]
    by_label = {}
    for name in LABELS:
        values: dict[str, list[float]] = {}
        for r in graded_rows:
            if name in r:
                values.setdefault(r[name], []).append(float(r["score"]))
        if values:
            by_label[name] = {
                k: {"n": len(v), "score": 100 * min(max(sum(v) / len(v), 0.0), 1.0)}
                for k, v in sorted(values.items())
            }
    chars = sorted(s["answer_chars"] for s in samples)
    return {
        "n_total": len(graded_rows),
        "n_samples": len(samples),
        "n_truncated": sum(bool(s.get("truncated")) for s in samples),
        "n_no_answer": sum(bool(s.get("no_answer")) for s in samples),
        "score": 100 * min(max(mean, 0.0), 1.0),
        # Of the unclipped mean; the clip only matters at the ends of the scale.
        "score_stderr": 100 * stderr if stderr is not None else None,
        "context": {
            "score_unadjusted": 100
            * min(max(sum(unadjusted) / max(len(unadjusted), 1), 0.0), 1.0),
            "median_answer_chars": chars[len(chars) // 2] if chars else 0,
            "mean_answer_chars": sum(chars) / len(chars) if chars else 0,
            "n_criteria_graded": sum(len(s["criteria"]) for s in samples),
            "judge_usage": usage,
            "by_label": by_label,
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
    headline = (
        "score",
        "score_stderr",
        "n_total",
        "n_samples",
        "n_no_answer",
        "n_truncated",
    )
    print(json.dumps({k: metrics[k] for k in headline}, indent=2))


def _finalize(output_rows: list[dict], args: argparse.Namespace, **extra) -> None:
    metrics = compute_metrics(output_rows)
    metrics["model"] = args.model_path
    metrics["split"] = args.split
    metrics["judge"] = args.judge
    metrics.update(extra)
    _write_outputs(output_rows, metrics, args)


def run_merge(args: argparse.Namespace) -> None:
    """Recombine per-shard predictions.jsonl files into one merged output."""
    from eval_state import merge_rows

    all_rows = merge_rows(args, load_tasks, checkers.read_jsonl, "id")
    _finalize(all_rows, args, n_shards=len(args.merge_shards))


def run_from_raw(args: argparse.Namespace) -> None:
    """Grade saved outputs against the split's task ids without querying a server."""
    rows = load_tasks(args)
    by_id = {r["id"]: r for r in checkers.read_jsonl(args.from_raw)}
    missing = [row["id"] for row in rows if row["id"] not in by_id]
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(rows)} task ids missing from {args.from_raw}"
        )

    async def run() -> list[dict]:
        sem = asyncio.Semaphore(args.judge_concurrency)
        jobs = []
        for row in rows:
            saved = by_id[row["id"]].get("samples") or []
            outputs = [
                {
                    "output": s.get("output", s.get("text")),
                    "finish_reason": s.get("finish_reason"),
                }
                for s in saved
            ]
            if not outputs:
                raise SystemExit(f"no samples saved for id={row['id']}")
            jobs.append(grade_row(args, row, outputs, sem))
        return list(await asyncio.gather(*jobs))

    _finalize(asyncio.run(run()), args, source=args.from_raw)


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
    done_rows = (
        read_predictions(os.path.join(args.log_dir, "predictions.jsonl"))
        if args.log_dir
        else []
    )
    expected = {row["id"] for row in rows}
    if any(row["id"] not in expected for row in done_rows):
        raise SystemExit(
            "Resume contains rows outside this shard; use a new output directory"
        )
    if any(r.get("n_samples") != args.n_samples for r in done_rows):
        raise SystemExit(
            "Saved predictions used a different --n-samples; use a new output directory"
        )
    done_ids = {r["id"] for r in done_rows}
    if done_ids:
        before = len(rows)
        rows = [r for r in rows if r["id"] not in done_ids]
        print(
            f">>> resuming: {before - len(rows)} row(s) already scored, "
            f"{len(rows)} remaining"
        )
    print(
        f">>> {len(rows)} task(s) (split={args.split}, shard {args.shard_index}/"
        f"{args.num_shards}, judge={args.judge}); querying server for {args.model_path}"
    )
    new_rows = asyncio.run(generate_and_grade_all(args, rows)) if rows else []
    _finalize(done_rows + new_rows, args)


if __name__ == "__main__":
    main()
