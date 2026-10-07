#!/usr/bin/env python3
"""Evaluate NormAd through an existing OpenAI-compatible inference server.

Write resumable predictions.jsonl and metrics.json with headline macro_f1;
secondary metrics appear under context. Split files already exclude neutral
rows with missing explanations.

Live mode queries concurrently and checkpoints every --checkpoint-every rows.
--from-raw rescores existing outputs by task ID without inference. Shard
launchers skip completed shards; live runs skip saved predictions.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import checkers
import prompt_building as pb
from eval_state import atomic_json, read_predictions, stamp_metrics
from openai import AsyncOpenAI

DATA_DIR = Path(os.environ.get("PDB_BENCHMARK_DIR", Path(__file__).resolve().parent))


def parse_args() -> argparse.Namespace:
    """Parse CLI args shared by the live, --from-raw, and --merge-shards run modes."""
    p = argparse.ArgumentParser(
        description="Evaluate a model on NormAd via a vLLM server."
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
    # Shard-merge mode: recombine already-graded predictions.jsonl files instead of
    # generating.
    p.add_argument("--merge-shards", type=str, nargs="+", default=None)
    p.add_argument(
        "--out-dir", type=str, default=None, help="Output dir for --merge-shards mode."
    )
    # Backfill mode: rescore an existing raw predictions.jsonl instead of querying a
    # server.
    p.add_argument(
        "--from-raw",
        type=str,
        default=None,
        help="Path to an existing predictions.jsonl (id/pred schema) to rescore "
        "against --split's task ids, instead of running live inference.",
    )
    return p.parse_args()


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


async def generate_and_score_all(
    args: argparse.Namespace, rows: list[dict]
) -> list[dict]:
    """Generate and score concurrently, checkpointing for row-level resume."""
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "tok")
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    extra_body: dict = {"min_p": args.min_p, "top_k": args.top_k}
    if args.enable_thinking is not None:
        extra_body["chat_template_kwargs"] = {"enable_thinking": args.enable_thinking}

    sem = asyncio.Semaphore(args.max_connections)
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
            completion = ""
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
                    if attempt == 3 or getattr(e, "status_code", None) in (
                        400,
                        401,
                        403,
                        404,
                    ):
                        rid = row["id"]
                        print(
                            (
                                "[warn] request for id="
                                f"{rid}"
                                " failed after retries: "
                                f"{type(e).__name__}"
                            )
                        )
                        raise RuntimeError(
                            "Evaluation request exhausted retries"
                        ) from e
                    else:
                        await asyncio.sleep(2 * (attempt + 1))

            out = _score_row(
                row,
                pred=checkers.parse_prediction(completion)
                if args.enable_thinking is False or "</think>" in completion
                else "",
                output=completion,
            )
            out["finish_reason"] = resp.choices[0].finish_reason
            out["truncated"] = resp.choices[0].finish_reason == "length"
            output_rows.append(out)
            pending.append(out)
            flush()

    try:
        outcomes = await asyncio.gather(*(one(r) for r in rows), return_exceptions=True)
    finally:
        flush(force=True)

    failures = [item for item in outcomes if isinstance(item, BaseException)]
    if failures:
        raise RuntimeError(
            (
                "Evaluation incomplete: "
                f"{len(failures)}"
                " request/grading failures; successful rows saved"
            )
        )
    return output_rows


def _score_row(row: dict, *, pred: str, output: str | None) -> dict:
    gold = checkers.decode_answer(row["gold_label"])
    if pred == "":
        reason = "unparseable"
    elif pred == gold:
        reason = "correct"
    else:
        reason = "incorrect"
    return {
        "id": row["id"],
        "country": row["country"],
        "axis": row["axis"],
        "question": row["task_text"],
        "gold": gold,
        "pred": pred,
        "output": output,
        "score": pred == gold,
        "correctness_reason": reason,
    }


def compute_metrics(graded_rows: list[dict]) -> dict:
    """Score graded rows; macro_f1 is the headline, everything else is context."""
    graded_rows = sorted(graded_rows, key=lambda row: str(row["id"]))
    n = len(graded_rows)
    n_unparseable = sum(1 for r in graded_rows if r["pred"] == "")
    s = checkers.score(graded_rows, checkers.LABELS)
    return {
        "n_total": n,
        "n_truncated": sum(bool(row.get("truncated")) for row in graded_rows),
        "n_unparseable_predictions": n_unparseable,
        # Headline: this benchmark is compared on Macro-F1 alone.
        "macro_f1": s["macro_f1"],
        "macro_f1_stderr": s.get("macro_f1_stderr"),
        # Secondary, developer-facing context only -- not used to rank models.
        "context": {
            "accuracy": s["accuracy"],
            "accuracy_stderr": s["accuracy_stderr"],
            "micro_f1": s["micro_f1"],
            "micro_f1_stderr": s.get("micro_f1_stderr"),
            "per_class_f1": s["per_class_f1"],
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

    headline_keys = (
        "macro_f1",
        "macro_f1_stderr",
        "n_total",
        "n_unparseable_predictions",
    )
    print(json.dumps({k: metrics[k] for k in headline_keys}, indent=2))


def _finalize(
    output_rows: list[dict], args: argparse.Namespace, **extra_metrics
) -> None:
    """Score output_rows, stamp on run metadata, and write predictions + metrics."""
    metrics = compute_metrics(output_rows)
    metrics["model"] = args.model_path
    metrics["split"] = args.split
    metrics.update(extra_metrics)
    _write_outputs(output_rows, metrics, args)


def run_merge(args: argparse.Namespace) -> None:
    """Recombine per-shard predictions.jsonl files into one merged output."""
    from eval_state import merge_rows

    all_rows = merge_rows(args, load_tasks, checkers.read_jsonl, "id")
    _finalize(all_rows, args, n_shards=len(args.merge_shards))


def run_from_raw(args: argparse.Namespace) -> None:
    """Rescore saved outputs against split task IDs without querying the server."""
    rows = load_tasks(args)
    entry_by_id = {r["id"]: r for r in checkers.read_jsonl(args.from_raw)}

    output_rows = []
    n_missing = 0
    for row in rows:
        entry = entry_by_id.get(row["id"])
        if entry is None:
            n_missing += 1
            output_rows.append(_score_row(row, pred="", output=None))
        else:
            # Legacy input files (pre-dating this pipeline's "output" field name)
            # store the raw completion under "raw"; newer ones may already say
            # "output" -- accept either.
            completion = entry.get("output", entry.get("raw"))
            output_rows.append(
                _score_row(
                    row,
                    pred=checkers.parse_prediction(completion)
                    if args.enable_thinking is False or "</think>" in completion
                    else "",
                    output=completion,
                )
            )
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

    if args.from_raw:
        if args.split is None:
            raise SystemExit("--split val|test is required with --from-raw")
        run_from_raw(args)
        return

    if args.split is None:
        raise SystemExit("--split val|test is required (unless using --merge-shards)")

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
            f">>> resuming: {n_before - len(rows)} row(s) already scored in "
            f"{args.log_dir}/predictions.jsonl, {len(rows)} remaining"
        )

    print(
        f">>> {len(rows)} task(s) (split={args.split}, shard {args.shard_index}/"
        f"{args.num_shards}); querying server for {args.model_path}"
    )
    new_output_rows = asyncio.run(generate_and_score_all(args, rows)) if rows else []

    output_rows = done_rows + new_output_rows
    _finalize(output_rows, args)


if __name__ == "__main__":
    main()
