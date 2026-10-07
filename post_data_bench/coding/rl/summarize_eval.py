"""Report abstention, pass, and pass-given-attempt rates with Wilson intervals.

These separate coding changes from abstention shifts hidden by pooled accuracy.
Usage: python rl/summarize_eval.py <predictions.jsonl> [more.jsonl ...]"""
import argparse
import json
import math
import os
import collections

IMPOSSIBLE = ("correct_abstention", "incorrect_on_impossible")
DEFAULT_TASKS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             ".hidden", "test_tasks.jsonl")


def shard_ids(tasks_file: str, num_shards: int, shard_index: int) -> set[str]:
    """Select IDs by task-file stride, since predictions arrive in completion order."""
    rows = [json.loads(l) for l in open(tasks_file) if l.strip()]
    return {r["question_id"] for r in rows[shard_index::num_shards]}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fmt(k: int, n: int) -> str:
    if n == 0:
        return f"{'--':>17}"
    lo, hi = wilson(k, n)
    return f"{k/n:>6.3f} [{lo:.3f},{hi:.3f}]"


def summarize(path: str, keep: set[str] | None = None, label: str | None = None) -> None:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if keep is not None:
        rows = [r for r in rows if r.get("question_id") in keep]
    if not rows:
        print(f"{path}: empty")
        return
    cr = collections.Counter(r.get("correctness_reason") for r in rows)
    imp = [r for r in rows if r.get("correctness_reason") in IMPOSSIBLE]
    pos = [r for r in rows if r.get("correctness_reason") not in IMPOSSIBLE]
    att = [r for r in pos if r.get("correctness_reason") != "false_abstention"]

    n_ab = cr["correct_abstention"]
    n_fa = cr["false_abstention"]
    n_pa = cr["passed"]
    acc = sum(r.get("score", 0) for r in rows) / len(rows)

    print(f"\n=== {label or path} ===")
    print(f"graded {len(rows)} rows  ({len(imp)} impossible, {len(pos)} possible, "
          f"prior {len(imp)/len(rows):.3f})")
    if "shard_" in path:
        print("note: one shard, and possibly still in progress -- resubmitting the "
              "eval extends it, since rows already here are skipped")
    print()
    print(f"  abstain | impossible   {fmt(n_ab, len(imp))}")
    print(f"  abstain | possible     {fmt(n_fa, len(pos))}   (false abstention)")
    print(f"  pass    | possible     {fmt(n_pa, len(pos))}")
    print(f"  pass    | ATTEMPTED    {fmt(n_pa, len(att))}   "
          f"(n={len(att)}; equals pass|possible when the model never falsely abstains)")
    print(f"\n  accuracy (this subset's prior)  {acc:.3f}")

    # Discrimination: does the model abstain more on impossible rows than possible
    # ones? A model that abstains uniformly scores z ~ 0 however high its
    # abstention rate is.
    if imp and pos:
        ai, ap = n_ab / len(imp), n_fa / len(pos)
        pool = (n_ab + n_fa) / len(rows)
        se = math.sqrt(pool * (1 - pool) * (1 / len(imp) + 1 / len(pos)))
        if se > 0:
            print(f"  discrimination z (abst|imp - abst|pos)  {(ai - ap) / se:+.2f}")

    trunc = sum(1 for r in rows if r.get("finish_reason") == "length"
                or r.get("truncated"))
    if trunc:
        print(f"  truncated at the token cap: {trunc}/{len(rows)} ({trunc/len(rows):.1%}) "
              f"-- these score 0 regardless")
    print(f"\n  categories: {dict(cr)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("predictions", nargs="+")
    ap.add_argument("--num-shards", type=int, default=None,
                    help="restrict to rows[shard_index::num_shards] of the task file, "
                         "the same subset rl/eval_rl_checkpoint.sh evaluates")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--tasks-file", default=DEFAULT_TASKS)
    a = ap.parse_args()
    keep = (shard_ids(a.tasks_file, a.num_shards, a.shard_index)
            if a.num_shards else None)
    if keep is not None:
        print(f"subset: shard {a.shard_index} of {a.num_shards} -> {len(keep)} rows "
              f"of {a.tasks_file}")
    for p in a.predictions:
        summarize(p, keep)
