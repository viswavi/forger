#!/usr/bin/env python3
"""Evaluate a model on the NormAd cultural-norm classification benchmark.

Same architecture as post_data_bench/coding/evaluate_lcb_v6_plus_impossible.py: talks
to an *already-running* vLLM OpenAI-compatible server (managed by evaluate_normad.sh)
at $VLLM_BASE_URL, fires every prompt concurrently (asyncio + AsyncOpenAI, bounded by
--max-connections), parses the {yes,no,neutral} answer, and writes:
  - predictions.jsonl  one row per instance: id, country, gold, pred, correct, raw
  - metrics.json        headline macro_f1 + secondary context metrics

Headline metric is macro_f1 -- the only number used to compare models on this
benchmark. accuracy/micro_f1/per_class_f1 are computed too but only ever surfaced
under metrics["context"], for developer inspection.

validation_tasks.jsonl / test_tasks.jsonl (built by make_splits.py) already have
neutral rows with no Explanation dropped -- this script never needs to re-apply that
filter.

Two run modes:
  - live (default): query a vLLM server for fresh completions.
  - --from-raw <predictions.jsonl>: skip inference and rescore an existing raw
    predictions file (id/pred schema, e.g. Qwen3_4b_normad_outputs/predictions.jsonl)
    against this split's task ids -- backfills metrics from already-computed outputs
    without paying for new inference.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import checkers
import prompt_building as pb
from openai import AsyncOpenAI

DATA_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Parse CLI args shared by the live, --from-raw, and --merge-shards run modes."""
    p = argparse.ArgumentParser(
        description="Evaluate a model on NormAd via a vLLM server."
    )
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
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--max-connections", type=int, default=128)
    p.add_argument(
        "--enable-thinking", dest="enable_thinking", action="store_true", default=None
    )
    p.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    p.add_argument("--json-output-file", type=str, default=None)
    p.add_argument("--log-dir", type=str, default=None)
    # Accepted-and-ignored: the sweep passes this to every eval script uniformly.
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
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
        name = "validation_tasks.jsonl" if args.split == "val" else "test_tasks.jsonl"
        tasks_file = str(DATA_DIR / name)
    rows = checkers.read_jsonl(tasks_file)
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows[args.shard_index :: args.num_shards]


async def generate_all(args: argparse.Namespace, rows: list[dict]) -> list[str]:
    """Fire one chat completion per row concurrently against the vLLM server."""
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "tok")
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    extra_body: dict = {"min_p": args.min_p, "top_k": args.top_k}
    if args.enable_thinking is not None:
        extra_body["chat_template_kwargs"] = {"enable_thinking": args.enable_thinking}

    sem = asyncio.Semaphore(args.max_connections)
    results: list[str] = [""] * len(rows)

    async def one(i: int, row: dict):
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
                    results[i] = resp.choices[0].message.content or ""
                    return
                except Exception as e:  # noqa: BLE001
                    if attempt == 3:
                        rid = row["id"]
                        print(f"[warn] request for id={rid} failed after retries: {e}")
                        results[i] = ""
                    else:
                        await asyncio.sleep(2 * (attempt + 1))

    await asyncio.gather(*(one(i, r) for i, r in enumerate(rows)))
    return results


def _score_row(row: dict, *, pred: str, raw: str | None) -> dict:
    gold = checkers.decode_answer(row["gold_label"])
    return {
        "id": row["id"],
        "country": row["country"],
        "gold": gold,
        "pred": pred,
        "correct": pred == gold,
        "raw": raw,
    }


def compute_metrics(graded_rows: list[dict]) -> dict:
    """Score graded rows; macro_f1 is the headline, everything else is context."""
    n = len(graded_rows)
    n_unparseable = sum(1 for r in graded_rows if r["pred"] == "")
    s = checkers.score(graded_rows, checkers.LABELS)
    return {
        "n_total": n,
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
        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=2)
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
    all_rows: list[dict] = []
    for path in args.merge_shards:
        shard_rows = checkers.read_jsonl(path)
        if not shard_rows:
            print(f"[merge] WARNING: {path} is missing or empty, skipping")
        all_rows.extend(shard_rows)
    print(
        f"[merge] merged {len(all_rows)} row(s) from {len(args.merge_shards)} shard(s)"
    )

    # Merge mode always writes to out_dir (or log_dir as a fallback), ignoring any
    # explicit --json-output-file -- matches how per-shard runs already resolve paths.
    args.log_dir = args.out_dir or args.log_dir or "."
    args.json_output_file = None
    _finalize(all_rows, args, n_shards=len(args.merge_shards))


def run_from_raw(args: argparse.Namespace) -> None:
    """Rescore an existing raw predictions.jsonl against this split's task ids.

    Reuses already-computed completions (matched by id) instead of querying a live
    server -- lets us backfill metrics.json for outputs generated before this pipeline
    existed, without paying for new inference.
    """
    rows = load_tasks(args)
    raw_by_id = {r["id"]: r for r in checkers.read_jsonl(args.from_raw)}

    output_rows = []
    n_missing = 0
    for row in rows:
        raw = raw_by_id.get(row["id"])
        if raw is None:
            n_missing += 1
            output_rows.append(_score_row(row, pred="", raw=None))
        else:
            output_rows.append(_score_row(row, pred=raw["pred"], raw=raw.get("raw")))
    if n_missing:
        print(
            f"[from-raw] WARNING: {n_missing}/{len(rows)} task id(s) missing "
            f"from {args.from_raw}"
        )

    _finalize(output_rows, args, source=args.from_raw)


def main() -> None:
    """Dispatch to the live, --from-raw, or --merge-shards run mode."""
    args = parse_args()

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
    print(
        f">>> {len(rows)} task(s) (split={args.split}, shard {args.shard_index}/"
        f"{args.num_shards}); querying server for {args.model_path}"
    )
    completions = asyncio.run(generate_all(args, rows))

    output_rows = [
        _score_row(row, pred=checkers.parse_prediction(completion), raw=completion)
        for row, completion in zip(rows, completions)
    ]
    _finalize(output_rows, args)


if __name__ == "__main__":
    main()
