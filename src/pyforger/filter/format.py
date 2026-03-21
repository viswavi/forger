"""Format-based filter operations for Forger.

Filters that remove instruction-tuning examples based on formatting criteria
such as length, invalid responses, and blocked keywords.
"""
from __future__ import annotations

from difflib import SequenceMatcher

from pyforger.core import DataSpec, NodeType

# Sourced from Eric Hartford's WizardLM cleaning script:
# https://huggingface.co/datasets/ehartford/WizardLM_alpaca_evol_instruct_70k_unfiltered
# Subset limited to AI self-reference and refusal patterns only.
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


def by_length(
    source: DataSpec,
    *,
    instruction_field: str,
    min_len: int = 10,
    max_len: int = 10000,
) -> DataSpec:
    """Filter items whose instruction word count is outside [min_len, max_len].

    Uses whitespace-delimited word count to determine length.

    Args:
        source: Upstream DataSpec providing list[dict] data.
        instruction_field: Key in each dict containing the instruction text
            (e.g. "instruction").
        min_len: Minimum allowed word count (inclusive).
        max_len: Maximum allowed word count (inclusive).

    Returns:
        A FILTER DataSpec that keeps only items within the length bounds.
    """

    def _execute(data: list[dict]) -> list[dict]:
        return [
            item
            for item in data
            if min_len <= len(item[instruction_field].split()) <= max_len
        ]

    return DataSpec(NodeType.FILTER, source, _execute, "filter_by_length")


def by_invalid_response(
    source: DataSpec,
    *,
    instruction_field: str,
    response_field: str,
    similarity_threshold: float = 0.85,
) -> DataSpec:
    """Filter items with invalid responses (copies of instruction or empty).

    Removes items where:
    - The response is None, empty, or whitespace-only.
    - The response exactly matches the instruction.
    - One string is a substring of the other and their lengths are within
      a factor of two (avoids false positives for short instructions).
    - The normalized string similarity exceeds the threshold.

    Args:
        source: Upstream DataSpec providing list[dict] data.
        instruction_field: Key containing the instruction text (e.g. "instruction").
        response_field: Key containing the response text (e.g. "response").
        similarity_threshold: SequenceMatcher ratio above which the response
            is considered a copy of the instruction.

    Returns:
        A FILTER DataSpec that keeps only items with valid responses.
    """

    def _is_invalid(item: dict) -> bool:
        response = item[response_field]
        if response is None or not response.strip():
            return True
        instruction = item[instruction_field]
        if response == instruction:
            return True
        shorter, longer = sorted([instruction, response], key=len)
        if len(longer) > 0 and len(shorter) / len(longer) > 0.5:
            if shorter in longer:
                return True
        ratio = SequenceMatcher(None, instruction, response).ratio()
        return ratio >= similarity_threshold

    def _execute(data: list[dict]) -> list[dict]:
        return [item for item in data if not _is_invalid(item)]

    return DataSpec(NodeType.FILTER, source, _execute, "filter_by_invalid_response")


def by_keyword_blocklist(
    source: DataSpec,
    *,
    response_field: str,
    blocklist: list[str] | None = None,
    case_sensitive: bool = False,
) -> DataSpec:
    """Filter items whose response contains blocked keywords.

    Uses substring matching against a configurable blocklist of phrases
    commonly found in AI refusals and disclaimers.

    Args:
        source: Upstream DataSpec providing list[dict] data.
        response_field: Key containing the response text (e.g. "response").
        blocklist: Keywords to block. Defaults to common AI refusal phrases.
        case_sensitive: Whether matching is case-sensitive.

    Returns:
        A FILTER DataSpec that keeps only items without blocked keywords.
    """
    keywords = blocklist if blocklist is not None else DEFAULT_BLOCKLIST

    def _execute(data: list[dict]) -> list[dict]:
        return [
            item
            for item in data
            if not _contains_blocked(item[response_field], keywords, case_sensitive)
        ]

    return DataSpec(NodeType.FILTER, source, _execute, "filter_by_keyword_blocklist")


def _contains_blocked(text: str, blocklist: list[str], case_sensitive: bool) -> bool:
    """Check whether text contains any keyword from the blocklist."""
    if not case_sensitive:
        text = text.lower()
        return any(kw.lower() in text for kw in blocklist)
    return any(kw in text for kw in blocklist)
