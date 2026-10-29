r"""Check judge.py against a FrontierCS checkout (developer tool, not for agents).

    python validate_judge.py --source SRC load        # every config loads
    python validate_judge.py --source SRC compile     # every checker/interactor builds
    python validate_judge.py --source SRC solutions --out results.jsonl \\
        [--models gemini3pro,gpt5] [--per-problem 1] [--problems 1,2,3]

``solutions`` judges files from algorithmic/solutions/<id>/<model>[_k].cpp and
appends one JSON line per file (resumable: files already in --out are skipped).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import judge


def _problems(src: Path, only: set[str] | None) -> list[Path]:
    root = src / "algorithmic/problems"
    dirs = sorted(
        (p for p in root.iterdir() if (p / "config.yaml").exists()),
        key=lambda p: int(p.name),
    )
    return [p for p in dirs if not only or p.name in only]


def cmd_load(args) -> None:
    """Load every config and report test counts, types, and missing files."""
    kinds: collections.Counter[str] = collections.Counter()
    n_cases = []
    for pdir in _problems(args.source, args.only):
        try:
            p = judge.load_problem(pdir)
        except Exception as e:  # noqa: BLE001
            print(f"LOAD-FAIL {pdir.name}: {e}")
            continue
        kinds[p.type] += 1
        n_cases.append(len(p.cases))
        missing = [
            c.input for c in p.cases if not (pdir / "testdata" / c.input).exists()
        ]
        if missing:
            print(f"MISSING {pdir.name}: {len(missing)} inputs, e.g. {missing[:3]}")
        name = p.interactor or p.checker
        if not (pdir / name).exists():
            print(f"NO-JUDGE {pdir.name}: {name}")
    n_cases.sort()
    print(
        f"types {dict(kinds)}; tests per problem min {n_cases[0]} "
        f"median {n_cases[len(n_cases) // 2]} max {n_cases[-1]} total {sum(n_cases)}"
    )


def _compile_one(pdir: str, testlib: str, cache: str) -> tuple[str, str | None]:
    try:
        judge.compiled_judge_program(
            judge.load_problem(pdir), Path(testlib), Path(cache)
        )
        return pdir, None
    except Exception as e:  # noqa: BLE001
        return pdir, str(e)[-800:]


def cmd_compile(args) -> None:
    """Compile every checker and interactor into the judge cache."""
    testlib = args.source / "algorithmic/judge/include"
    start = time.monotonic()
    failures = 0
    with ProcessPoolExecutor(args.jobs) as pool:
        futures = [
            pool.submit(_compile_one, str(p), str(testlib), str(args.cache))
            for p in _problems(args.source, args.only)
        ]
        for f in as_completed(futures):
            pdir, err = f.result()
            if err:
                failures += 1
                print(f"COMPILE-FAIL {Path(pdir).name}: {err}")
    print(
        f"compiled {len(futures) - failures}/{len(futures)} in "
        f"{time.monotonic() - start:.0f}s"
    )


def _judge_file(
    pdir: str, solution: str, testlib: str, cache: str, workers: int
) -> dict:
    start = time.monotonic()
    try:
        result = judge.judge(
            pdir,
            Path(solution).read_text(errors="replace"),
            testlib_dir=testlib,
            cache_dir=cache,
            workers=workers,
        )
    except Exception:  # noqa: BLE001
        result = {"status": "judge_error", "error": traceback.format_exc()[-1500:]}
    statuses = collections.Counter(c["status"] for c in result.get("cases", []))
    signals = collections.Counter(
        m.group(1)
        for c in result.get("cases", [])
        for m in [re.search(r"signal=(\d+)", c["msg"])]
        if m
    )
    return {
        "signals": dict(signals),
        "tle_cpu": sorted(
            c["time"]
            for c in result.get("cases", [])
            if c["status"] == "Time Limit Exceeded"
        )[:5],
        "problem": Path(pdir).name,
        "solution": Path(solution).name,
        "status": result["status"],
        "score": result.get("score"),
        "score_unbounded": result.get("score_unbounded"),
        "passed": result.get("passed"),
        "n_cases": len(result.get("cases", [])),
        "case_statuses": dict(statuses),
        "seconds": round(time.monotonic() - start, 1),
        "message": (result.get("message") or result.get("error") or "")[-600:],
    }


def cmd_solutions(args) -> None:
    """Judge published solutions and append one JSON line per file."""
    testlib = args.source / "algorithmic/judge/include"
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["problem"], r["solution"]))
    models = set(args.models.split(",")) if args.models else None
    jobs = []
    for pdir in _problems(args.source, args.only):
        sol_dir = args.source / "algorithmic/solutions" / pdir.name
        if not sol_dir.is_dir():
            continue
        files = sorted(f for f in sol_dir.iterdir() if f.suffix == ".cpp")
        per_model: collections.Counter[str] = collections.Counter()
        for f in files:
            # gpt5_3.cpp is gpt5's 4th attempt; gpt5_low.cpp is another model.
            base, _, k = f.stem.rpartition("_")
            model = base if k.isdigit() else f.stem
            if models and model not in models:
                continue
            if per_model[model] >= args.per_problem:
                continue
            per_model[model] += 1
            if (pdir.name, f.name) not in done:
                jobs.append((str(pdir), str(f)))
    print(
        f"{len(jobs)} solutions to judge ({len(done)} already in {args.out})",
        flush=True,
    )
    start = time.monotonic()
    with ProcessPoolExecutor(args.jobs) as pool, open(args.out, "a") as out:
        futures = [
            pool.submit(_judge_file, p, s, str(testlib), str(args.cache), args.workers)
            for p, s in jobs
        ]
        for i, f in enumerate(as_completed(futures), 1):
            out.write(json.dumps(f.result()) + "\n")
            out.flush()
            if i % 50 == 0:
                print(f"{i}/{len(jobs)} in {time.monotonic() - start:.0f}s", flush=True)
    print(f"done {len(jobs)} in {time.monotonic() - start:.0f}s")


def main() -> None:
    """Parse arguments and run one mode."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--source", type=Path, required=True)
    p.add_argument(
        "--cache",
        type=Path,
        default=Path(os.environ.get("TMPDIR", "/tmp")) / "fcs-judge-cache",
    )
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--problems", default=None, help="comma-separated problem ids")
    p.add_argument("--models", default=None)
    p.add_argument(
        "--per-problem", type=int, default=1, help="files per model per problem"
    )
    p.add_argument("--out", type=Path, default=Path("judge_validation.jsonl"))
    p.add_argument("mode", choices=["load", "compile", "solutions"])
    args = p.parse_args()
    args.only = set(args.problems.split(",")) if args.problems else None
    {"load": cmd_load, "compile": cmd_compile, "solutions": cmd_solutions}[args.mode](
        args
    )


if __name__ == "__main__":
    main()
