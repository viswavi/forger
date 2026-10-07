"""Summarize per-process rollout logs by row type and abstention rate.

Usage: python rl/summarize_rollouts.py <log-dir> [--bucket 2000] [--hacked]
When step IDs are unavailable, group timestamp-ordered records into buckets;
256 records match one step at the default rollout settings.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

CORRECT = ("passed", "correct_abstention")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this utility."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("log_dir", help="directory of rollouts_*.jsonl")
    p.add_argument(
        "--bucket",
        type=int,
        default=256,
        help="records per bucket (default 256 = one rollout step at "
        "rollout-batch-size 32 x n-samples-per-prompt 8)",
    )
    p.add_argument(
        "--hacked",
        action="store_true",
        help="print completions the verbatim-test guard rejected",
    )
    p.add_argument(
        "--csv",
        metavar="PATH",
        help="write per-training-step reward curves to PATH (see step_rows)",
    )
    p.add_argument(
        "--train-log",
        metavar="PATH",
        help="the run's .out; step boundaries are then read from its timestamped "
        "'step N:' lines instead of being inferred from record gaps. Exact, "
        "and the only way to key a run whose records predate the `step` field.",
    )
    p.add_argument(
        "--gap",
        type=float,
        default=120.0,
        help="seconds of silence that separate two rollout steps (default 120). "
        "Only used when records carry no rollout_id; the training pause "
        "between rollouts runs minutes against a ~2s median inter-record gap.",
    )
    return p.parse_args()


def load(log_dir: Path) -> list[dict]:
    """Load persisted rollout rows grouped by training step."""
    rows: list[dict] = []
    files = sorted(log_dir.glob("rollouts_*.jsonl"))
    if not files:
        raise SystemExit(f"no rollouts_*.jsonl under {log_dir}")
    for f in files:
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # a torn final line from a killed writer
    rows.sort(key=lambda r: r.get("ts", 0))
    print(f"{len(rows)} records from {len(files)} writer file(s)\n")
    return rows


def summarize(rows: list[dict], label: str) -> None:
    """Compute aggregate metrics from saved evaluation or rollout results."""
    if not rows:
        return
    n = len(rows)
    imp = [r for r in rows if r.get("impossible")]
    pos = [r for r in rows if not r.get("impossible")]
    acc = sum(r["reward"] for r in rows) / n
    # Compare base_reward with benchmark accuracy; reward includes training shaping.
    base = sum(r.get("base_reward", r["reward"]) for r in rows) / n

    def rate(sub, pred):
        return (sum(1 for r in sub if pred(r)) / len(sub)) if sub else float("nan")

    parts = [f"{label:>12s}  n={n:5d}  reward={acc:.3f}"]
    if abs(base - acc) > 1e-9:
        parts[0] += f" (metric={base:.3f})"
    if imp:
        parts.append(
            (
                "| impossible n="
                f"{len(imp):4d}"
                " abstain="
                f"{rate(imp, lambda r: r['abstained']):.3f}"
            )
        )
    if pos:
        parts.append(
            (
                "| possible n="
                f"{len(pos):4d}"
                " pass="
                f"{rate(pos, lambda r: r['category'] == 'passed'):.3f}"
                " false_abst="
                f"{rate(pos, lambda r: r['category'] == 'false_abstention'):.3f}"
            )
        )
    print("  ".join(parts))


def is_eval_record(r: dict) -> bool:
    """Identify validation records: training samples carry group_index, evals do not."""
    return r.get("group_index") is None


def steps_from_train_log(path: str) -> list[float]:
    """Read step timestamps from stdout as upper bounds for preceding rollouts."""
    import datetime
    import re

    pat = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\][^\n]*? - step (\d+):")
    seen: dict[int, float] = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                ts = datetime.datetime.strptime(
                    m.group(1), "%Y-%m-%d %H:%M:%S"
                ).timestamp()
                seen.setdefault(int(m.group(2)), ts)
    return [seen[k] for k in sorted(seen)]


def assign_steps(
    rows: list[dict], gap: float, train_log: str | None = None
) -> list[list[dict]]:
    """Bucket by training step, falling back to pauses between generation periods."""
    for key in ("step", "rollout_id"):
        if rows and rows[0].get(key) is not None:
            buckets: dict = {}
            for r in rows:
                buckets.setdefault(r[key], []).append(r)
            return [buckets[k] for k in sorted(buckets)]

    if train_log:
        bounds = steps_from_train_log(train_log)
        if bounds:
            bounded_steps: list[list[dict]] = [[] for _ in range(len(bounds) + 1)]
            for r in rows:
                i = 0
                while i < len(bounds) and r["ts"] > bounds[i]:
                    i += 1
                bounded_steps[i].append(r)
            return [s for s in bounded_steps if s]
        print(f"  warning: no 'step N:' lines in {train_log}; using gap segmentation")

    steps: list[list[dict]] = [[]]
    for i, r in enumerate(rows):
        if i and r["ts"] - rows[i - 1]["ts"] > gap:
            steps.append([])
        steps[-1].append(r)
    return [s for s in steps if s]


def step_rows(steps: list[list[dict]]) -> list[dict]:
    """Report reward and benchmark metric curves for each training step.

    GRPO optimizes reward; metric is the strict 0/1 benchmark score. They differ
    where rewards are shaped, including false abstentions, so metric is the
    comparable number.
    """
    out = []
    for i, rows in enumerate(steps):
        imp = [r for r in rows if r.get("impossible")]
        pos = [r for r in rows if not r.get("impossible")]
        mean = lambda xs: (sum(xs) / len(xs)) if xs else ""  # noqa: E731
        toks = sorted(r["response_tokens"] for r in rows if r.get("response_tokens"))
        out.append(
            {
                "step": i,
                "n": len(rows),
                "n_possible": len(pos),
                "n_impossible": len(imp),
                "reward_mean": mean([r["reward"] for r in rows]),
                "reward_possible": mean([r["reward"] for r in pos]),
                "reward_impossible": mean([r["reward"] for r in imp]),
                "metric_mean": mean([r.get("base_reward", r["reward"]) for r in rows]),
                "abstain_impossible": mean([float(r["abstained"]) for r in imp]),
                "false_abstain_possible": mean(
                    [float(r["category"] == "false_abstention") for r in pos]
                ),
                "pass_possible": mean([float(r["category"] == "passed") for r in pos]),
                "median_tokens": toks[len(toks) // 2] if toks else "",
                "trunc_frac": mean(
                    [
                        float(r["truncated"])
                        for r in rows
                        if r.get("truncated") is not None
                    ]
                ),
                "partial": 0,
            }
        )
    # Flag partial steps: shorter abstentions finish first, biasing their rate upward.
    sizes = sorted(r["n"] for r in out)
    if len(out) > 1:
        typical = sizes[len(sizes) // 2]
        if out[-1]["n"] < 0.9 * typical:
            out[-1]["partial"] = 1
    return out


def main() -> None:
    """Run the command-line utility described in the module documentation."""
    args = parse_args()
    rows = load(Path(args.log_dir))

    eval_rows = [r for r in rows if is_eval_record(r)]
    rows = [r for r in rows if not is_eval_record(r)]
    if eval_rows:
        print(
            f"({len(eval_rows)} validation records held out of the training curves; "
            f"{len(rows)} training samples remain)\n"
        )

    # A run that has only reached its first eval has no training rows to divide by.
    # Report the validation pass and stop, leaving any existing csv alone.
    if not rows:
        if eval_rows:
            print("=== validation (no training steps yet) ===")
            for i, st in enumerate(assign_steps(eval_rows, args.gap, args.train_log)):
                summarize(st, f"eval {i}")
        if args.csv:
            print(f"\nno completed training steps yet -- left {args.csv} untouched")
        return

    steps = assign_steps(rows, args.gap, args.train_log)
    print(
        f"=== by training step ({len(steps)} steps"
        + (
            ""
            if (
                args.train_log
                or (
                    rows
                    and (
                        rows[0].get("step") is not None
                        or rows[0].get("rollout_id") is not None
                    )
                )
            )
            else f", segmented on gaps > {args.gap:g}s"
        )
        + ") ==="
    )
    srows_for_flags = step_rows(steps)
    for i, st in enumerate(steps):
        summarize(
            st, f"step {i}" + (" (partial)" if srows_for_flags[i]["partial"] else "")
        )
    if eval_rows:
        print("\n=== validation (held out of the curves above) ===")
        for i, st in enumerate(assign_steps(eval_rows, args.gap, args.train_log)):
            summarize(st, f"eval {i}")

    if args.csv:
        import csv as _csv

        srows = step_rows(steps)
        with open(args.csv, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(srows[0].keys()))
            w.writeheader()
            w.writerows(srows)
        print(f"\nwrote {len(srows)} step rows -> {args.csv}")

    print("\n=== by bucket (chronological) ===")
    for i in range(0, len(rows), args.bucket):
        summarize(rows[i : i + args.bucket], f"[{i // args.bucket}]")

    print("\n=== overall ===")
    summarize(rows, "all")

    cats = Counter(r["category"] for r in rows)
    print("\ncategories:", json.dumps(dict(cats.most_common()), indent=2))

    # Gaming the process reward looks like doubt-word counts inflating over
    # training while the abstention rate stays flat, so track both.
    shaped = [r for r in rows if r.get("shaped")]
    if shaped:
        print(
            f"\nprocess reward paid on {len(shaped)}/{len(rows)} "
            f"({len(shaped) / len(rows):.1%}) of completions"
        )
    doubt = [r["doubt_matches"] for r in rows if r.get("doubt_matches") is not None]
    if doubt:
        di = [
            r["doubt_matches"]
            for r in rows
            if r.get("impossible") and r.get("doubt_matches") is not None
        ]
        dp = [
            r["doubt_matches"]
            for r in rows
            if not r.get("impossible") and r.get("doubt_matches") is not None
        ]
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")  # noqa: E731
        print(
            f"doubt matches  median impossible={med(di)}  possible={med(dp)}"
            f"   (a rising median with a flat abstention rate is the tell that "
            f"the shaping is being gamed)"
        )
    truncation_known = [r for r in rows if r.get("truncated") is not None]
    if truncation_known:
        nt = sum(bool(r["truncated"]) for r in truncation_known)
        print(
            (
                "response-cap truncation: "
                f"{nt}"
                "/"
                f"{len(truncation_known)}"
                " ("
                f"{nt / len(truncation_known):.1%}"
                ")"
            )
        )
    else:
        print("response-cap truncation: unknown (legacy logs lack sample status)")

    hacked = [r for r in rows if r.get("hacked")]
    print(
        (
            "\nverbatim-test guard fired on "
            f"{len(hacked)}"
            "/"
            f"{len(rows)}"
            " ("
            f"{len(hacked) / len(rows):.2%}"
            "); this heuristic misses reward hacks and can flag valid "
            "code. A low rate does not establish correctness on unseen "
            "tests."
        )
    )
    if args.hacked:
        for r in hacked[:20]:
            print(f"\n--- {r['label']}\n{(r.get('code') or '')[:600]}")

    lens = [r["response_tokens"] for r in rows if r.get("response_tokens")]
    if lens:
        lens.sort()
        print(
            f"\nresponse tokens: median={lens[len(lens) // 2]} "
            f"p95={lens[int(len(lens) * 0.95)]} max={lens[-1]}"
        )


if __name__ == "__main__":
    main()
