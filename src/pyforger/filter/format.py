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
    min_len: int = 10,
    max_len: int = 10000,
    instruction_field: str = "instruction",
) -> DataSpec:
    """Filter items whose instruction length is outside [min_len, max_len].

    Uses character count to determine length.

    Args:
        source: Upstream DataSpec providing list[dict] data.
        min_len: Minimum allowed character count (inclusive).
        max_len: Maximum allowed character count (inclusive).
        instruction_field: Key in each dict containing the instruction text.

    Returns:
        A FILTER DataSpec that keeps only items within the length bounds.
    """

    def _execute(data: list[dict]) -> list[dict]:
        return [
            item
            for item in data
            if min_len <= len(item[instruction_field]) <= max_len
        ]

    return DataSpec(NodeType.FILTER, source, _execute, "filter_by_length")


def by_invalid_response(
    source: DataSpec,
    *,
    instruction_field: str = "instruction",
    output_field: str = "output",
    similarity_threshold: float = 0.85,
) -> DataSpec:
    """Filter items with invalid responses (copies of instruction or empty).

    Removes items where:
    - The output is None, empty, or whitespace-only.
    - The output exactly matches the instruction.
    - One string is a substring of the other and their lengths are within
      a factor of two (avoids false positives for short instructions).
    - The normalized string similarity exceeds the threshold.

    Args:
        source: Upstream DataSpec providing list[dict] data.
        instruction_field: Key containing the instruction text.
        output_field: Key containing the output/response text.
        similarity_threshold: SequenceMatcher ratio above which the output
            is considered a copy of the instruction.

    Returns:
        A FILTER DataSpec that keeps only items with valid responses.
    """

    def _is_invalid(item: dict) -> bool:
        output = item[output_field]
        if output is None or not output.strip():
            return True
        instruction = item[instruction_field]
        if output == instruction:
            return True
        shorter, longer = sorted([instruction, output], key=len)
        if len(longer) > 0 and len(shorter) / len(longer) > 0.5:
            if shorter in longer:
                return True
        ratio = SequenceMatcher(None, instruction, output).ratio()
        return ratio >= similarity_threshold

    def _execute(data: list[dict]) -> list[dict]:
        return [item for item in data if not _is_invalid(item)]

    return DataSpec(NodeType.FILTER, source, _execute, "filter_by_invalid_response")


def by_keyword_blocklist(
    source: DataSpec,
    *,
    blocklist: list[str] | None = None,
    output_field: str = "output",
    case_sensitive: bool = False,
) -> DataSpec:
    """Filter items whose output contains blocked keywords.

    Uses substring matching against a configurable blocklist of phrases
    commonly found in AI refusals and disclaimers.

    Args:
        source: Upstream DataSpec providing list[dict] data.
        blocklist: Keywords to block. Defaults to common AI refusal phrases.
        output_field: Key containing the output/response text.
        case_sensitive: Whether matching is case-sensitive.

    Returns:
        A FILTER DataSpec that keeps only items without blocked keywords.
    """
    raw_keywords = blocklist if blocklist is not None else DEFAULT_BLOCKLIST
    if not case_sensitive:
        normalized_keywords = [kw.lower() for kw in raw_keywords]
    else:
        normalized_keywords = list(raw_keywords)

    def _execute(data: list[dict]) -> list[dict]:
        return [
            item
            for item in data
            if not _contains_blocked(
                item[output_field], normalized_keywords, case_sensitive
            )
        ]

    return DataSpec(NodeType.FILTER, source, _execute, "filter_by_keyword_blocklist")


def _contains_blocked(text: str, blocklist: list[str], case_sensitive: bool) -> bool:
    """Check whether text contains any keyword from the blocklist.

    Expects blocklist keywords to already be lowered when case_sensitive is False.
    """
    if not case_sensitive:
        text = text.lower()
    return any(kw in text for kw in blocklist)
