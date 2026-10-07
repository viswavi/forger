"""Measure nonzero GRPO reward variance by task type before training.

Sample n completions per prompt at rollout temperature and grade with the RL
verifier. Equal-reward groups contribute no gradient, so a base model that
never abstains may provide no signal on impossible rows.

Usage: python rl/measure_group_variance.py --model-path <checkpoint>
           --data rl/data/full/train.jsonl --no-thinking --limit 64 -n 8
Run in the trl environment with one GPU.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

CODING_DIR = Path(__file__).resolve().parent.parent
if str(CODING_DIR) not in sys.path:
    sys.path.insert(0, str(CODING_DIR))

import checkers  # noqa: E402
import prompt_building as pb  # noqa: E402
from rl.impossible_lcb_rm import _drain_pool, _grade_async  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this utility."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--data", default=str(CODING_DIR / "rl/data/full/train.jsonl"))
    p.add_argument("--limit", type=int, default=64, help="prompts sampled from --data")
    p.add_argument("-n", "--n-samples", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--out", default=None, help="write the per-group table here")
    p.add_argument(
        "--generations-cache",
        default=None,
        help="Reuse (or write) raw completions here. Generation is the expensive "
        "half and grading is the fragile one, so keep them separable: a grading "
        "failure then costs a re-grade, not another sampling pass.",
    )
    return p.parse_args()


def main() -> None:
    """Run the command-line utility described in the module documentation."""
    import random

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    args = parse_args()
    max_tokens = args.max_tokens or (3072 if args.no_thinking else 20480)

    rows = checkers.read_jsonl(args.data)
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit]

    cache = Path(args.generations_cache) if args.generations_cache else None
    import hashlib
    from argparse import Namespace

    from eval_state import atomic_json, identity

    cache_identity = {
        name: getattr(args, name)
        for name in (
            "model_path",
            "n_samples",
            "temperature",
            "top_p",
            "no_thinking",
            "seed",
            "limit",
        )
    }
    cache_identity["max_tokens"] = max_tokens
    cache_identity["rows_sha256"] = hashlib.sha256(
        json.dumps(rows, sort_keys=True).encode()
    ).hexdigest()
    cache_identity["model_signature"] = identity(Namespace(model_path=args.model_path))[
        "model_signature"
    ]
    meta = Path(str(cache) + ".meta.json") if cache else None
    if cache and cache.exists():
        if (
            meta is None
            or not meta.is_file()
            or json.loads(meta.read_text()) != cache_identity
        ):
            raise SystemExit(
                "Generation cache configuration or data changed; choose a new cache"
            )
        cached = {c["label"]: c["completions"] for c in checkers.read_jsonl(cache)}
        completions = [cached[r["label"]] for r in rows]
        print(f"reusing {len(completions)} cached generations from {cache}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        chat_kwargs = {"enable_thinking": False} if args.no_thinking else {}
        prompts = [
            tokenizer.apply_chat_template(
                r["prompt"], tokenize=False, add_generation_prompt=True, **chat_kwargs
            )
            for r in rows
        ]
        llm = LLM(
            model=args.model_path,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=8192 + max_tokens,
            enforce_eager=False,
        )
        outputs = llm.generate(
            prompts,
            SamplingParams(
                n=args.n_samples,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=max_tokens,
                seed=args.seed,
            ),
        )
        completions = [[c.text for c in o.outputs] for o in outputs]
        if cache:
            checkers.write_jsonl(
                cache,
                [
                    {"label": r["label"], "completions": c}
                    for r, c in zip(rows, completions, strict=True)
                ],
            )
            atomic_json(meta, cache_identity)
            print(f"wrote generations -> {cache}")
        # vLLM holds the GPU and a good deal of host memory; the grading pool is
        # subprocesses, so release the engine before spawning them.
        del llm

    async def grade_all() -> list[list[dict]]:
        async def one_group(row, texts):
            return await asyncio.gather(
                *(
                    _grade_async(
                        {**row["metadata"], "thinking": not args.no_thinking},
                        t,
                        row["label"],
                    )
                    for t in texts
                )
            )

        try:
            return await asyncio.gather(
                *(one_group(r, c) for r, c in zip(rows, completions, strict=True))
            )
        finally:
            await _drain_pool()

    graded = asyncio.run(grade_all())

    table = []
    for row, group in zip(rows, graded, strict=True):
        gold = checkers.decode_answer(row["metadata"]["gold_response"])
        scores = [g["score"] for g in group]
        table.append(
            {
                "label": row["label"],
                "impossible": gold == pb.ABSTENTION_MARKER,
                "scores": scores,
                "mean": statistics.fmean(scores),
                "nonzero_std": len(set(scores)) > 1,
                "n_abstained": sum(g["abstained"] for g in group),
                "categories": dict(Counter(g["category"] for g in group)),
            }
        )

    by_type: dict[str, list[dict]] = defaultdict(list)
    for t in table:
        by_type["impossible" if t["impossible"] else "possible"].append(t)

    print(f"\n=== {args.model_path}")
    print(
        f"    thinking={'off' if args.no_thinking else 'on'} "
        f"temperature={args.temperature} n={args.n_samples} prompts={len(table)}\n"
    )
    for kind in (
        "possible",
        "impossible",
    ):
        rows_k = by_type[kind]
        if not rows_k:
            continue
        usable = sum(t["nonzero_std"] for t in rows_k)
        print(
            f"  {kind:11s} n={len(rows_k):4d}  "
            f"mean reward={statistics.fmean(t['mean'] for t in rows_k):.3f}  "
            f"groups with signal={usable}/{len(rows_k)} ({usable / len(rows_k):.1%})"
        )
        if kind == "impossible":
            abst = sum(t["n_abstained"] for t in rows_k)
            total = sum(len(t["scores"]) for t in rows_k)
            print(
                (
                    "              abstained on "
                    f"{abst}"
                    "/"
                    f"{total}"
                    " samples ("
                    f"{abst / total:.1%}"
                    ")"
                )
            )
    usable_all = sum(t["nonzero_std"] for t in table)
    print(
        (
            "\n  overall      mean reward="
            f"{statistics.fmean((t['mean'] for t in table)):.3f}"
            "  groups with signal="
            f"{usable_all}"
            "/"
            f"{len(table)}"
            " ("
            f"{usable_all / len(table):.1%}"
            ")"
        )
    )
    cats: Counter[str] = Counter()
    for t in table:
        cats.update(t["categories"])
    print(f"  categories   {json.dumps(dict(cats))}\n")

    if args.out:
        checkers.write_jsonl(args.out, table)
        print(f"  per-group table -> {args.out}")


if __name__ == "__main__":
    main()
