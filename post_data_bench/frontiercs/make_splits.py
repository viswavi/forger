"""Build FrontierCS's validation/secret-test split of the algorithmic problems.

Problems come from a FrontierCS checkout pinned to one commit. Problem 61 is left
out, as FrontierCS leaves it out of every reported mean (its output format changed).
A fixed-seed random sample of the rest becomes validation, in the ratio 50 : 207;
the others are the secret test set. Each split gets its task rows and a copy of its
problem directories (statement, config.yaml, checker or interactor, testdata/):

    validation_tasks.jsonl            problems/<id>/
    .hidden/test_tasks.jsonl          .hidden/problems/<id>/
    testlib/testlib.h                 problem_files.json (sha256 of every copied file)

Statements are masked so they do not name the contest site they came from (see
mask_source): agents read the validation statements, and the site's contest pages
would point them at the test problems and their published solutions.

prompt_building.SYSTEM_PROMPT is FrontierCS's CPP_SYSTEM_PROMPT (algorithmic/scripts/
generate_solutions.py), and checkers.extract_code its extract_cpp_code rule; those
modules do not name FrontierCS because agent sandboxes get copies of them.

Usage: python make_splits.py --source /path/to/Frontier-CS [--output-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
from pathlib import Path

import checkers
import judge
import prompt_building as pb
from split_contract import staged_build, write_manifest

DATA_DIR = Path(__file__).resolve().parent
SEED = 20261029
VAL_RATIO = 50 / 257
EXCLUDED = {"61": "output format changed; excluded from FrontierCS's own means"}
PINNED_COMMIT = "12ab26840cff15f1d0ab09004dafd07f92c14b60"


def _problem_ids(problems_root: Path) -> list[str]:
    """Every problem directory name, in numeric order."""
    return sorted(
        (p.name for p in problems_root.iterdir() if (p / "config.yaml").exists()),
        key=int,
    )


# Lines that only link to the site's pages, tools, or social-media threads.
_SOURCE_LINE = re.compile(r"atcoder\.jp|twitter", re.IGNORECASE)
# What may not survive masking: the site, its contest ids, social media.
_SOURCE_TRACE = re.compile(r"atcoder|\bahc\d|twitter", re.IGNORECASE)
MASKED_NAME = "Algoria"


def mask_source(statement: str) -> str:
    """The statement without its source site, e.g. "AtCoder's CEO" -> "Algoria's CEO".

    Lines linking to the site (visualizers, tool downloads, example pages) or to
    social media are dropped; they are of no use to a model without a browser.
    """
    lines = [line for line in statement.split("\n") if not _SOURCE_LINE.search(line)]
    masked = re.sub(r"AtCoder", MASKED_NAME, "\n".join(lines), flags=re.IGNORECASE)
    if _SOURCE_TRACE.search(masked):
        raise ValueError(
            f"source trace survives masking: {_SOURCE_TRACE.search(masked)}"
        )
    return masked


def _task_row(pdir: Path) -> dict:
    """One task row: the (masked) statement, its prompt text, and judging facts."""
    problem = judge.load_problem(pdir)
    statement = mask_source((pdir / "statement.txt").read_text())
    return {
        "id": pdir.name,
        "problem_id": pdir.name,
        "type": "interactive" if problem.interactor else "default",
        "n_tests": len(problem.cases),
        "statement": statement,
        "task_text": pb.build_user_content(statement),
    }


def _copy_problem(src: Path, dst: Path) -> None:
    """Copy the files judging needs: statement, config, checker/interactor, tests."""
    problem = judge.load_problem(src)
    dst.mkdir(parents=True)
    names = ["config.yaml", problem.interactor or problem.checker]
    for name in names:
        shutil.copy(src / name, dst / name)
    statement = mask_source((src / "statement.txt").read_text())
    (dst / "statement.txt").write_text(statement)
    shutil.copytree(src / "testdata", dst / "testdata")


def _digest_files(root: Path, subdirs: list[str]) -> dict[str, str]:
    """sha256 of every file under the given subdirectories, by relative path."""
    out = {}
    for sub in subdirs:
        for path in sorted((root / sub).rglob("*")):
            if path.is_file():
                out[str(path.relative_to(root))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    return out


def split_ids(ids: list[str]) -> tuple[list[str], list[str]]:
    """Fixed-seed random validation sample (VAL_RATIO of the problems) and the rest."""
    n_val = round(len(ids) * VAL_RATIO)
    val = sorted(random.Random(SEED).sample(ids, n_val), key=int)
    test = [i for i in ids if i not in set(val)]
    return val, test


def main() -> None:
    """Build and write the split, its problem copies, and the manifest."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="FrontierCS checkout at the pinned commit",
    )
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace an existing secret test split.",
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if (out / ".hidden/test_tasks.jsonl").exists() and not args.force:
        parser.error("test split already exists; use a new --output-dir or --force")

    commit = subprocess.run(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if commit != PINNED_COMMIT:
        raise SystemExit(f"FrontierCS checkout is at {commit}, not {PINNED_COMMIT}")
    problems_root = args.source / "algorithmic/problems"
    ids = [i for i in _problem_ids(problems_root) if i not in EXCLUDED]
    val_ids, test_ids = split_ids(ids)
    print(
        f"[splits] FrontierCS@{commit[:7]}: {len(ids)} problems "
        f"(excluded {sorted(EXCLUDED)}); val {len(val_ids)}, test {len(test_ids)}"
    )

    val_rows = [_task_row(problems_root / i) for i in val_ids]
    test_rows = [_task_row(problems_root / i) for i in test_ids]
    # staged_build publishes only manifest files, so the problem directories are
    # copied in place first; problem_files.json (in the manifest) pins their bytes.
    for sub in ("problems", ".hidden/problems", "testlib"):
        shutil.rmtree(out / sub, ignore_errors=True)
    for i in val_ids:
        _copy_problem(problems_root / i, out / "problems" / i)
    for i in test_ids:
        _copy_problem(problems_root / i, out / ".hidden/problems" / i)
    (out / "testlib").mkdir(parents=True)
    shutil.copy(
        args.source / "algorithmic/judge/include/testlib.h", out / "testlib/testlib.h"
    )
    files = _digest_files(out, ["problems", ".hidden/problems", "testlib"])
    with staged_build(out) as stage:
        checkers.write_jsonl(stage / "validation_tasks.jsonl", val_rows)
        checkers.write_jsonl(stage / ".hidden/test_tasks.jsonl", test_rows)
        digest = stage / "problem_files.json"
        digest.write_text(json.dumps(files, indent=1) + "\n")
        write_manifest(
            stage,
            [
                stage / "validation_tasks.jsonl",
                stage / ".hidden/test_tasks.jsonl",
                digest,
            ],
            val_rows,
            test_rows,
            seed=SEED,
            inputs={"frontiercs_commit": commit, "excluded": EXCLUDED},
        )
    print(f"[splits] wrote {out}")


if __name__ == "__main__":
    main()
