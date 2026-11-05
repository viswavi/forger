"""Remove training rows that overlap with the held-out test problems.

The sandbox never holds test text: the launcher writes only hashed 13-word windows
of each test problem's statement (`decontamination_signatures.json`, next to this
module); short statements such as a two-sentence situation use 8-word windows
(`SHORT`). A row overlaps a test problem when its prompt contains at least
`FRACTION` of that problem's windows (`LONG_FRACTION` for long statements) and at
least the signature's `min_matches`. Windows found in `COMMON` or more statements,
counting the validation statements too, are boilerplate (e.g. a site's standard
definition of lexicographic order) and ignored. Training data goes through
`filter_rows` before every training run, and only the number of removed rows is
reported.

  python -m rl.decontaminate --data rows.jsonl   # how many rows overlap
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

N = 13
FRACTION = 0.25
MIN_MATCHES = 10
COMMON = 3
# Window size and minimum matches for short statements (median 37 words): copies of
# half a test situation are caught, and 3 of 280 validation situations, near-copies
# of test ones, are flagged.
SHORT = (8, 6)
# Fraction for long statements (median 700 words, 13-word windows): a copied fifth
# of a statement is caught (162 of 179), while unrelated programming statements
# reach at most 0.03 of a statement's windows and validation statements 0.11 (their
# shared scoring boilerplate).
LONG_FRACTION = 0.15
# The statement field of each task's rows, and its length class.
STATEMENT_FIELDS = {
    "question_content": "default",
    "situation": "short",
    "statement": "long",
}
SIGNATURES = Path(__file__).with_name("decontamination_signatures.json")


def _words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _windows(text, n=N):
    """Hashed n-word windows of `text`, e.g. 2 for a 14-word statement and n=13."""
    words = _words(text)
    return {
        hashlib.sha256(" ".join(words[i : i + n]).encode()).hexdigest()[:16]
        for i in range(len(words) - n + 1)
    }


def build_signatures(
    statements, background=(), n=N, min_matches=MIN_MATCHES, fraction=FRACTION
):
    """Signatures of the test problem statements (no text, no ids, no order).

    `background` statements (e.g. the validation tasks) only count toward which
    windows are boilerplate; they are not matched themselves.
    """
    per_problem = [_windows(s, n) for s in statements]
    counts = Counter(w for windows in per_problem for w in windows)
    counts.update(w for s in background for w in _windows(s, n))
    index = {}
    sizes = []
    for number, windows in enumerate(per_problem):
        kept = [w for w in windows if counts[w] < COMMON]
        sizes.append(len(kept))
        for window in kept:
            index.setdefault(window, []).append(number)
    return {
        "n": n,
        "min_matches": min_matches,
        "fraction": fraction,
        "sizes": sizes,
        "windows": index,
    }


def _statements(path):
    """Each row's statement, and the length class of that field."""
    rows = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    field = next(f for f in STATEMENT_FIELDS if rows and f in rows[0])
    return [row[field] for row in rows], STATEMENT_FIELDS[field]


def benchmark_signatures(benchmark_dir):
    """Signatures from a benchmark build: its hidden test and its validation tasks."""
    root = Path(benchmark_dir)
    test, kind = _statements(root / ".hidden" / "test_tasks.jsonl")
    background, _ = _statements(root / "validation_tasks.jsonl")
    n, min_matches = SHORT if kind == "short" else (N, MIN_MATCHES)
    fraction = LONG_FRACTION if kind == "long" else FRACTION
    return build_signatures(test, background, n, min_matches, fraction)


@lru_cache(maxsize=1)
def signatures():
    """Load the sandbox's signature file, or build it from the host's test set."""
    if SIGNATURES.is_file():
        return json.loads(SIGNATURES.read_text())
    root = os.environ.get("PDB_BENCHMARK_DIR")
    if root and Path(root, ".hidden", "test_tasks.jsonl").is_file():
        return benchmark_signatures(root)
    return None


def _prompt_text(row):
    prompt = row["prompt"]
    if isinstance(prompt, str):
        return prompt
    return "\n".join(m["content"] for m in prompt if isinstance(m, dict))


def overlaps(row, signature):
    """Whether `row`'s prompt contains enough of one test problem's windows."""
    hits = Counter()
    for window in _windows(_prompt_text(row), signature["n"]):
        for problem in signature["windows"].get(window, ()):
            hits[problem] += 1
    sizes = signature["sizes"]
    minimum = signature.get("min_matches", MIN_MATCHES)
    fraction = signature.get("fraction", FRACTION)
    return any(
        count >= max(minimum, fraction * sizes[problem])
        for problem, count in hits.items()
    )


def filter_rows(rows):
    """Rows that do not overlap the test problems, and how many were removed.

    Without signatures (no test set configured) every row is kept and the count
    is None.
    """
    signature = signatures()
    if signature is None:
        return list(rows), None
    kept = [row for row in rows if not overlaps(row, signature)]
    return kept, len(rows) - len(kept)


def main():
    """Report how many rows of a dataset overlap the held-out test problems."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="JSON lines with a prompt field")
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.data).read_text().splitlines()
        if line.strip()
    ]
    kept, removed = filter_rows(rows)
    if removed is None:
        print("no test signatures available; nothing checked")
    else:
        print(f"{removed} of {len(rows)} rows overlap the held-out test problems")


if __name__ == "__main__":
    main()
