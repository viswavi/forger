"""Compare two saved pipeline runs: compare_pipelines.

Every finished @pipeline run is saved with each step's cache key and a
digest of each part of it (code, each global, each argument). Two runs that
share their first steps have the same keys for those steps; the first step
whose key differs is where they diverge, and the differing parts say why, e.g.
solve: argument prompt differs. The steps' outputs are read back from the
step cache, so rows can be matched across runs.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Any

from pyforger import _stats
from pyforger._tracking import STEP_OP_PREFIX, PipelineRecord, StepRecord
from pyforger.inspect.runs import _mean_text, _score_values
from pyforger.op import _cache_dir, _read_result
from pyforger.steps import saved_runs

_EXAMPLES = 3  # rows shown per side of a row comparison
_PREVIEW_CHARS = 70
_SIDE = {"only_a": "only in A", "only_b": "only in B"}
# A run saved before steps recorded their cache keys: whether the step changed
# is unknown, so it is never the first difference.
_NO_KEY = "no key recorded (an older run)"
_CACHE_KEY = re.compile(r"[0-9a-f]{64}")


def _resolve(
    all_runs: list[PipelineRecord], ref: str | PipelineRecord | None, before: int
) -> PipelineRecord | None:
    """The run ``ref`` names (a run id, or a pipeline's latest) before ``before``."""
    if isinstance(ref, PipelineRecord):
        return ref
    pool = all_runs[:before]
    for rec in reversed(pool):
        if ref is None or rec.run_id == ref or rec.name == ref:
            return rec
    return None


def _pick(
    a: str | PipelineRecord | None, b: str | PipelineRecord | None
) -> tuple[PipelineRecord, PipelineRecord]:
    all_runs = saved_runs()
    if isinstance(a, PipelineRecord) and isinstance(b, PipelineRecord):
        return a, b
    run_b = _resolve(all_runs, b, len(all_runs))
    if run_b is None:
        raise KeyError(f"no saved run matches {b!r}; see pyforger.inspect.runs()")
    before = next(
        (i for i, r in enumerate(all_runs) if r.run_id == run_b.run_id),
        len(all_runs),
    )
    if a is None:  # the latest earlier run of the same kind (smoke or full)
        same_kind = [r for r in all_runs[:before] if r.smoke_rows == run_b.smoke_rows]
        run_a = same_kind[-1] if same_kind else None
    else:
        run_a = _resolve(all_runs, a, before) or _resolve(all_runs, a, len(all_runs))
    if run_a is None or run_a.run_id == run_b.run_id:
        raise KeyError(
            f"no earlier saved run matches {a!r}; see pyforger.inspect.runs()"
        )
    return run_a, run_b


def _keyed_steps(rec: PipelineRecord) -> dict[tuple[str, int], StepRecord]:
    """Steps by (name, occurrence), e.g. a step called in a loop."""
    seen: dict[str, int] = defaultdict(int)
    out: dict[tuple[str, int], StepRecord] = {}
    for step in rec.steps:
        out[(step.name, seen[step.name])] = step
        seen[step.name] += 1
    return out


def _why(a: StepRecord, b: StepRecord) -> list[str]:
    """What differs between two runs of a step, from their key parts."""
    if a.cache_key and a.cache_key == b.cache_key:
        return []
    if not a.key_parts or not b.key_parts:
        return [_NO_KEY]
    out = []
    for part in dict.fromkeys([*a.key_parts, *b.key_parts]):
        if part not in a.key_parts:
            out.append(f"{part} only in B")
        elif part not in b.key_parts:
            out.append(f"{part} only in A")
        elif a.key_parts[part] != b.key_parts[part]:
            out.append(f"{part} differs")
    return out or ["key differs"]


def _counts(step: StepRecord | None) -> str:
    if step is None:
        return "-"
    n_in = "-" if step.n_in is None else f"{step.n_in:,}"
    n_out = "-" if step.n_out is None else f"{step.n_out:,}"
    return f"{n_in} → {n_out}"


def _op_yields(step: StepRecord) -> dict[str, tuple[int, int]]:
    """Rows in and out per op name, summed over repeated calls."""
    out: dict[str, list[int]] = {}
    for call in step.ops:
        if call.n_in is None or call.n_out is None or call.error is not None:
            continue
        entry = out.setdefault(call.op_name, [0, 0])
        entry[0] += call.n_in
        entry[1] += call.n_out
    return {name: (n_in, n_out) for name, (n_in, n_out) in out.items()}


def _verdict(clear: bool) -> str:
    return "clearly different" if clear else "within noise"


def _yield_diff(
    ya: tuple[int, int] | None, yb: tuple[int, int] | None
) -> dict[str, Any] | None:
    """B's yield minus A's with its 95% interval; ``clear`` when it excludes 0."""
    if not ya or not yb or not ya[0] or not yb[0]:
        return None
    if ya[1] > ya[0] or yb[1] > yb[0]:  # not a share of the input (fan-out)
        return None
    lo, hi = _stats.diff_interval(yb[1], yb[0], ya[1], ya[0])
    return {
        "diff": yb[1] / yb[0] - ya[1] / ya[0],
        "ci": (lo, hi),
        "clear": lo > 0 or hi < 0,
    }


def _signed(x: float) -> str:
    """``+0.25``, ``-0.10``; ``+0.00`` rather than ``-0.00``."""
    text = f"{x:+.2f}"
    return "+0.00" if text == "-0.00" else text


def _mean_diff(a: list[float] | None, b: list[float] | None) -> bool | None:
    """Whether two score means clearly differ (their difference's 95% interval)."""
    sa = _stats.mean_interval(a or [])
    sb = _stats.mean_interval(b or [])
    if sa is None or sb is None:
        return None
    return abs(sb[0] - sa[0]) > (sa[1] ** 2 + sb[1] ** 2) ** 0.5


def _step_output(step: StepRecord) -> Any:
    """The step's cached output, or None when it is not in the cache.

    Like an op's cache read, only a file of this user is loaded.
    """
    if not step.cache_key or not _CACHE_KEY.fullmatch(step.cache_key):
        return None
    safe = "".join(
        c if c.isalnum() or c in "._-" else "_" for c in STEP_OP_PREFIX + step.name
    )
    path = _cache_dir() / safe / f"{step.cache_key}.pkl"
    try:
        if hasattr(os, "getuid") and path.stat().st_uid != os.getuid():
            return None
        return _read_result(path, _cache_dir())
    except Exception:
        return None


def _row_key(rows_a: list[Any], rows_b: list[Any], key: str | None) -> str | None:
    """A field that identifies rows in both lists: ``key``, ``id``, or a unique one."""
    rows = rows_a + rows_b
    if not rows or not all(isinstance(r, dict) for r in rows):
        return None
    if key is not None:
        return key if all(key in r for r in rows) else None
    shared = set(rows[0])
    for r in rows:
        shared &= set(r)
    ordered = sorted(shared, key=lambda f: (f != "id", not f.endswith("id"), f))
    for field in ordered:
        for side in (rows_a, rows_b):
            values = [r[field] for r in side]
            if not all(isinstance(v, (str, int)) for v in values):
                break
            if len(set(values)) != len(values):
                break
        else:
            return field
    return None


def _preview(row: Any) -> str:
    text = json.dumps(row, default=str) if not isinstance(row, str) else row
    return text if len(text) <= _PREVIEW_CHARS else text[: _PREVIEW_CHARS - 1] + "…"


def _compare_rows(
    a: StepRecord, b: StepRecord, key: str | None
) -> dict[str, Any] | None:
    out_a, out_b = _step_output(a), _step_output(b)
    if not isinstance(out_a, list) or not isinstance(out_b, list):
        return None
    field = _row_key(out_a, out_b, key)
    if field is not None:
        ids_a = {r[field]: r for r in out_a}
        ids_b = {r[field]: r for r in out_b}
    else:  # whole rows
        ids_a = {json.dumps(r, sort_keys=True, default=str): r for r in out_a}
        ids_b = {json.dumps(r, sort_keys=True, default=str): r for r in out_b}
    only_a = [ids_a[k] for k in ids_a if k not in ids_b]
    only_b = [ids_b[k] for k in ids_b if k not in ids_a]
    # Rows both runs have (by key) whose values differ, e.g. another solution.
    changed: dict[str, int] = defaultdict(int)
    examples: list[tuple[Any, Any]] = []
    for k in ids_a:
        if k not in ids_b or field is None:
            continue
        row_a, row_b = ids_a[k], ids_b[k]
        fields = [
            f
            for f in dict.fromkeys([*row_a, *row_b])
            if json.dumps(row_a.get(f), sort_keys=True, default=str)
            != json.dumps(row_b.get(f), sort_keys=True, default=str)
        ]
        for f in fields:
            changed[f] += 1
        if fields and len(examples) < _EXAMPLES:
            examples.append(
                ({f: row_a.get(f) for f in fields}, {f: row_b.get(f) for f in fields})
            )
    return {
        "matched_by": field or "whole row",
        "both": sum(1 for k in ids_a if k in ids_b),
        "only_a": only_a,
        "only_b": only_b,
        "changed": dict(changed),
        "changed_examples": examples,
    }


def compare_pipelines(
    a: str | PipelineRecord | None = None,
    b: str | PipelineRecord | None = None,
    *,
    key: str | None = None,
) -> dict[str, Any]:
    """Print where two saved pipeline runs diverge, and how their steps differ.

    ``a`` and ``b`` are run ids (from ``runs()``) or pipeline names (that
    pipeline's latest run); by default ``b`` is the latest run and ``a`` the
    run before it. E.g. after running ``main_a`` and then ``main_b``, which
    share ``find_problems`` and differ in ``solve``'s prompt::

        compare_pipelines("main_a", "main_b")

    prints, per step, both runs' rows in → out, yield, and cost, and why the
    step's cache keys differ (``argument prompt differs``, ``global PROMPT
    differs``, ``code differs``; a step whose only difference is its input
    rows inherited it from an earlier step). For each step that differs it
    lists each op's yield in both runs and, when both outputs are still in the
    step cache, which output rows only one run kept, matched by ``key`` (by
    default an ``id`` field, or another field unique in both).

    Returns a dict with ``a`` / ``b`` (run ids), ``first_difference`` (a
    step name or None), and ``steps``: one dict per step with ``name``,
    ``same``, ``why``, and, for steps that differ, ``ops`` and ``rows``.
    """
    run_a, run_b = _pick(a, b)
    steps_a, steps_b = _keyed_steps(run_a), _keyed_steps(run_b)
    order = list(dict.fromkeys([*steps_a, *steps_b]))
    print(
        f"compare A = {run_a.run_id} ({run_a.name}, {run_a.started_at}) "
        f"→ B = {run_b.run_id} ({run_b.name}, {run_b.started_at})"
    )
    width = max([len(name) for name, _ in order] + [4])
    print(
        f"  {'step':<{width}}  {'A in → out':>14}  {'cost':>7}  "
        f"{'B in → out':>14}  {'cost':>7}  difference"
    )
    result: dict[str, Any] = {
        "a": run_a.run_id,
        "b": run_b.run_id,
        "first_difference": None,
        "steps": [],
    }
    details = []
    for name, i in order:
        sa, sb = steps_a.get((name, i)), steps_b.get((name, i))
        label = name if i == 0 else f"{name}#{i + 1}"
        if sa is None or sb is None:
            why = ["only in A" if sb is None else "only in B"]
        else:
            why = _why(sa, sb)
        row_parts = [p for p in why if p.startswith("argument ")]
        only_rows = (
            bool(why)
            and len(row_parts) == len(why)
            and all(
                sa is not None
                and sb is not None
                and _is_rows_part(sa, p.split()[1].strip())
                for p in row_parts
            )
        )
        text = (
            "same"
            if not why
            else (
                "input rows differ (from an earlier step)"
                if only_rows
                else "; ".join(why)
            )
        )
        cost_a = f"${sa.cost_usd:.2f}" if sa else "-"
        cost_b = f"${sb.cost_usd:.2f}" if sb else "-"
        print(
            f"  {label:<{width}}  {_counts(sa):>14}  {cost_a:>7}  "
            f"{_counts(sb):>14}  {cost_b:>7}  {text}"
        )
        entry: dict[str, Any] = {"name": label, "same": not why, "why": why}
        unknown = why == [_NO_KEY]
        if why and result["first_difference"] is None and not (only_rows or unknown):
            result["first_difference"] = label
        if why and sa is not None and sb is not None:
            details.append((label, sa, sb, entry))
        result["steps"].append(entry)
    out_a = "?" if run_a.n_out is None else f"{run_a.n_out:,}"
    out_b = "?" if run_b.n_out is None else f"{run_b.n_out:,}"
    print(
        f"  output: A {out_a} rows (${run_a.cost_usd:.2f} this run), "
        f"B {out_b} rows (${run_b.cost_usd:.2f} this run)"
    )
    if result["first_difference"]:
        print(f"  first difference: {result['first_difference']}")
    for label, sa, sb, entry in details:
        print(f"\n  {label}:")
        yields_a, yields_b = _op_yields(sa), _op_yields(sb)
        entry["ops"] = {}
        for op_name in dict.fromkeys([*yields_a, *yields_b]):
            ya, yb = yields_a.get(op_name), yields_b.get(op_name)
            diff = _yield_diff(ya, yb)
            entry["ops"][op_name] = {"a": ya, "b": yb, "b_minus_a": diff}
            if (ya and ya[0] != ya[1]) or (yb and yb[0] != yb[1]):
                text = f"    {op_name}: A {_yield_text(ya)}  B {_yield_text(yb)}"
                if diff is not None:
                    lo, hi = diff["ci"]
                    text += (
                        f"  · B − A {_signed(diff['diff'])} (95% CI "
                        f"{_signed(lo)} to {_signed(hi)}): {_verdict(diff['clear'])}"
                    )
                print(text)
        scores_a, scores_b = _score_values(sa), _score_values(sb)
        entry["scores"] = {}
        for label_score in dict.fromkeys([*scores_a, *scores_b]):
            va, vb = scores_a.get(label_score), scores_b.get(label_score)
            clear = _mean_diff(va, vb)
            entry["scores"][label_score] = {"clearly_different": clear}
            text = f"    mean {label_score}: A {_mean_text(va)}  B {_mean_text(vb)}"
            if clear is not None:
                text += f"  · {_verdict(clear)}"
            print(text)
        rows = _compare_rows(sa, sb, key)
        entry["rows"] = rows
        if rows is None:
            print("    rows: not in the step cache (or not lists); rerun to compare")
            continue
        n_a, n_b = len(rows["only_a"]), len(rows["only_b"])
        print(
            f"    rows (matched by {rows['matched_by']}): {rows['both']:,} in both, "
            f"{n_a:,} only in A, {n_b:,} only in B"
        )
        if n_a + n_b:
            # Rows one run kept and the other did not: if both runs kept rows
            # equally often, a fair coin would split them.
            p = _stats.sign_test(n_a, n_b)
            rows["sign_test_p"] = p
            lead = (
                f"{'A' if n_a > n_b else 'B'} keeps more of them"
                if n_a != n_b
                else "an even split"
            )
            print(
                f"      {n_a:,} vs {n_b:,}: {_verdict(p < 0.05)} "
                f"({lead}; sign test {_stats.fmt_p(p)})"
            )
        for side in ("only_a", "only_b"):
            for row in rows[side][:_EXAMPLES]:
                print(f"      {_SIDE[side]}: {_preview(row)}")
        if rows["changed"]:
            fields = ", ".join(f"{f} in {n:,}" for f, n in rows["changed"].items())
            print(f"    rows in both whose values differ: {fields}")
            for in_a, in_b in rows["changed_examples"]:
                print(f"      A: {_preview(in_a)}")
                print(f"      B: {_preview(in_b)}")
    return result


def _is_rows_part(step: StepRecord, param: str) -> bool:
    """True when ``param`` is the step's rows argument (its input rows)."""
    if step.inputs:
        return param in {s.name for s in step.inputs}
    return step.n_in is not None and param in _first_params(step)


def _first_params(step: StepRecord) -> list[str]:
    params = [p.split(" ", 1)[1] for p in step.key_parts if p.startswith("argument ")]
    return params[:1]


def _yield_text(counts: tuple[int, int] | None) -> str:
    if counts is None:
        return "-"
    n_in, n_out = counts
    share = f" ({n_out / n_in:.2f})" if n_in else ""
    return f"{n_out:,}/{n_in:,}{share}"
