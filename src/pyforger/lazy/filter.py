"""Filter ops (lazy). Remove items based on criteria."""
from __future__ import annotations

from pyforger.lazy.core import op


@op(name="filter.by_length")
def by_length(
    items: list[dict],
    *,
    field: str,
    min_len: int = 1,
    max_len: int = 10_000,
) -> list[dict]:
    """Keep items whose ``field`` word-count is within [min_len, max_len]."""
    out = []
    for item in items:
        val = item.get(field, "")
        n_words = len(val.split()) if isinstance(val, str) else 0
        if min_len <= n_words <= max_len:
            out.append(item)
    return out


@op(name="filter.by_keyword_blocklist")
def by_keyword_blocklist(
    items: list[dict],
    *,
    field: str,
    blocklist: list[str],
) -> list[dict]:
    """Drop items whose ``field`` contains any blocklisted token (word-level)."""
    blocked = {b.lower() for b in blocklist}
    out = []
    for item in items:
        text = item.get(field, "")
        tokens = {t.lower() for t in text.split()} if isinstance(text, str) else set()
        if not (tokens & blocked):
            out.append(item)
    return out


@op(name="filter.by_exact_agreement")
def by_exact_agreement(
    items: list[dict],
    *,
    group_by: str,
    agree_on: str,
) -> list[dict]:
    """Within each group, keep items only if all siblings agree on ``agree_on``."""
    from collections import defaultdict

    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[item.get(group_by, "")].append(item)

    out = []
    for group in groups.values():
        values = {item.get(agree_on) for item in group}
        if len(values) == 1:
            out.extend(group)
    return out


@op(name="filter.by_invalid_response")
def by_invalid_response(
    items: list[dict],
    *,
    instruction_field: str,
    response_field: str,
) -> list[dict]:
    """Drop items with empty or obviously-degenerate response text."""
    out = []
    for item in items:
        instr = item.get(instruction_field, "")
        resp = item.get(response_field, "")
        if not resp or not isinstance(resp, str):
            continue
        if resp.strip().lower() in {"", "n/a", "none", "null"}:
            continue
        if instr and resp.strip() == instr.strip():
            continue
        out.append(item)
    return out


@op(name="filter.by_rouge_l")
def by_rouge_l(
    items: list[dict],
    *,
    field: str,
    threshold: float = 0.7,
) -> list[dict]:
    """Greedy ROUGE-L dedup: drop items too similar to any already-kept one."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError("rouge-score required for by_rouge_l") from None

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    kept: list[dict] = []
    kept_texts: list[str] = []
    for item in items:
        text = item.get(field, "")
        if any(scorer.score(other, text)["rougeL"].fmeasure >= threshold for other in kept_texts):
            continue
        kept.append(item)
        kept_texts.append(text)
    return kept
