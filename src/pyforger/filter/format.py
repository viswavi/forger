"""Format-based filter operations.

Length, invalid-response, and keyword-blocklist filters.
"""
from __future__ import annotations

from difflib import SequenceMatcher

from pyforger import _tracking
from pyforger.op import op
from pyforger.utils import get_field

# From Eric Hartford's WizardLM cleaning script:
# https://huggingface.co/datasets/ehartford/WizardLM_alpaca_evol_instruct_70k_unfiltered
DEFAULT_BLOCKLIST = [
    "as an ai",
    "as a language model",
    "as an ai language model",
    "as a large language model",
    "i cannot fulfill your request",
    "i cannot assist",
    "i cannot provide",
    "i'm sorry, but as an ai",
    "i'm sorry, i cannot",
    "my programming",
    "against my programming",
    "i am an ai",
    "i'm an ai",
    "i am here to assist",
    "my purpose is to",
    "my knowledge cutoff",
]


@op(name="filter.by_length")
def by_length(
    items: list[dict],
    *,
    instruction_field: str,
    min_len: int = 10,
    max_len: int = 10000,
) -> list[dict]:
    """Keep items whose ``instruction_field`` word count is in [min_len, max_len]."""
    word_counts: list[int] = []
    kept: list[dict] = []
    for item in items:
        words = len(get_field(item, instruction_field, "by_length").split())
        word_counts.append(words)
        if min_len <= words <= max_len:
            kept.append(item)
    _tracking.record_scores(
        "words", word_counts, keep="between", lower=min_len, upper=max_len
    )
    return kept


@op(name="filter.by_invalid_response")
def by_invalid_response(
    items: list[dict],
    *,
    instruction_field: str,
    response_field: str,
    similarity_threshold: float = 0.85,
) -> list[dict]:
    """Filter items with invalid responses (copies of instruction or empty).

    Removes items where:
    - The response is None, empty, or whitespace-only.
    - The response exactly matches the instruction.
    - One string is a substring of the other and their lengths are within
      a factor of two (avoids false positives for short instructions).
    - The normalized string similarity exceeds the threshold.
    """

    def _is_invalid(item: dict) -> bool:
        response = get_field(item, response_field, "by_invalid_response")
        if response is None or not response.strip():
            return True
        instruction = get_field(item, instruction_field, "by_invalid_response")
        if response == instruction:
            return True
        shorter, longer = sorted([instruction, response], key=len)
        if len(longer) > 0 and len(shorter) / len(longer) > 0.5:
            if shorter in longer:
                return True
        ratio = SequenceMatcher(None, instruction, response).ratio()
        return ratio >= similarity_threshold

    return _tracking.keep_where(items, [not _is_invalid(item) for item in items])


@op(name="filter.by_keyword_blocklist")
def by_keyword_blocklist(
    items: list[dict],
    *,
    response_field: str,
    blocklist: list[str] | None = None,
    case_sensitive: bool = False,
) -> list[dict]:
    """Drop items whose ``response_field`` contains any blocklisted phrase."""
    keywords = blocklist if blocklist is not None else DEFAULT_BLOCKLIST
    keep = [
        not _contains_blocked(
            get_field(item, response_field, "by_keyword_blocklist"),
            keywords,
            case_sensitive,
        )
        for item in items
    ]
    return _tracking.keep_where(items, keep)


def _contains_blocked(text: str, blocklist: list[str], case_sensitive: bool) -> bool:
    if not case_sensitive:
        text = text.lower()
        return any(kw.lower() in text for kw in blocklist)
    return any(kw in text for kw in blocklist)
