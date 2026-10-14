"""Building block for conditional LM generation (conditioned on upstream input)."""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any, Union

from pyforger.acquire._model_utils import _build_prompt, _parse_output, _reasoning_for
from pyforger.acquire.model import Model, Prompt, split_reasoning
from pyforger.op import op
from pyforger.utils import get_field

PromptTemplate = Union[str, Callable[[dict], Prompt]]

_NUMBER = r"-?[0-9][0-9,]*(?:\.[0-9]+)?"
# A number written with thousands separators, e.g. "-1,234,567.5".
_GROUPED_NUMBER_RE = re.compile(r"-?[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?")

# Answer-extraction patterns tried in order, most explicit first. Each captures
# the final-answer token from a free-form response; the last is a trailing-number
# fallback, applied only if none of the explicit markers match.
_ANSWER_PATTERNS = [
    re.compile(rf"ANSWER\s*[:=]\s*\$?\\?\(?\$?\s*({_NUMBER})", re.I),
    re.compile(rf"\\boxed\{{\s*\$?\s*({_NUMBER})\s*\$?\s*\}}"),
    re.compile(rf"####\s*({_NUMBER})"),
    re.compile(rf"final answer\s*(?:is|:|=)?\s*\$?({_NUMBER})", re.I),
    re.compile(_NUMBER),
]


def _get_logger() -> logging.Logger:
    """Lazy logger to avoid circular import with utils."""
    from pyforger.utils import get_formatted_logger as _gf

    return _gf(__name__)


def _resolve_prompt(template: PromptTemplate, item: dict) -> Prompt:
    """Resolve a prompt template against one input item.

    A string template is formatted with the item's keys (e.g. ``"{text}"``
    is replaced by ``item["text"]``). A callable is called with the item.
    """
    if isinstance(template, str):
        return template.format_map(item)
    return template(item)


@op(name="alter.generate_from_context")
def generate_from_context(
    items: list[dict],
    prompt_fn: PromptTemplate,
    model: Model,
    *,
    samples_per_item: int = 1,
    output_schema: dict[str, Any] | None = None,
    temperature: float = 1.0,
    max_retries: int = 3,
    batch_size: int = 10,
) -> list[dict | None]:
    r"""Generate LLM outputs conditioned on upstream input items.

    ``prompt_fn`` wraps each input item in a prompt which is passed to ``model`` for
    generation. ``items`` must be a single list of dicts; to condition on multiple
    upstream sources, combine them first with ``{**dict1, **dict2}`` or similar.

    Args:
        items: Input rows passed one-by-one to ``prompt_fn``. To combine multiple
            upstream sources, merge items from dictionaries before passing here.
        prompt_fn: A format string (keys filled from the input item dict) or a
            callable ``(dict) -> Prompt``. Callables may return a string or a
            list of chat messages in OpenAI format.
        model: Model instance used for generation.
        samples_per_item: Number of responses to generate per input item.
        output_schema: Optional dict of expected keys. When set, responses are
            JSON-parsed and validated; failures are retried. When None, each
            response is stored as ``{"text": ...}``.
        temperature: Sampling temperature passed to the model.
        max_retries: Additional attempts for prompts whose output fails to parse.
        batch_size: Number of prompts per ``model.generate_batch()`` call.

    Returns:
        List of length ``len(items) * samples_per_item`` in input order, grouped
        by item (item ``i``'s samples occupy slots ``i * samples_per_item`` to
        ``(i + 1) * samples_per_item``). Each slot holds the parsed output dict,
        or ``None`` if the prompt failed to parse after ``max_retries``. When the
        model returns a reasoning trace separately in `GenerateResult.reasoning`,
        it is included under the `thinking_trace` key. Inline `<think>` traces (e.g.
        from `LocalModel`, which supplies `None` for that metadata) are not extracted;
        use `split_reasoning` to separate them out as needed.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if samples_per_item <= 0:
        raise ValueError("samples_per_item must be positive")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    log = _get_logger()

    prompts: list[Prompt] = [
        _build_prompt(_resolve_prompt(prompt_fn, item), output_schema) for item in items
    ]

    results: list[dict | None] = [None] * (len(prompts) * samples_per_item)

    # In order to handle retries while generating multiple samples per item via the n=
    # parameter of generate_batch, we keep a list of pending prompts with the number of
    # samples still needed for each, so we can track the amount of computation still
    # required (up to max_retries).
    pending: list[tuple[Prompt, list[int]]] = [
        (p, list(range(i * samples_per_item, (i + 1) * samples_per_item)))
        for i, p in enumerate(prompts)
    ]

    for attempt in range(1 + max_retries):
        if not pending:
            break
        next_pending: list[tuple[Prompt, list[int]]] = []

        # generate_batch uses one n per call, so group prompts needing the same
        # number of completions (on the first pass all need samples_per_item).
        by_count: dict[int, list[tuple[Prompt, list[int]]]] = {}
        for entry in pending:
            by_count.setdefault(len(entry[1]), []).append(entry)

        for n, group in by_count.items():
            for start in range(0, len(group), batch_size):
                batch = group[start : start + batch_size]
                batch_results = model.generate_batch(
                    [p for p, _ in batch], temperature=temperature, n=n
                )
                if len(batch_results) != len(batch):
                    log.warning(
                        "generate_batch returned %d results for %d prompts; %s",
                        len(batch_results),
                        len(batch),
                        "giving up" if attempt == max_retries else "will retry",
                    )
                    next_pending.extend(batch)
                    continue
                for (prompt, slots), completions in zip(batch, batch_results):
                    unfilled = slots[len(completions) :]  # short batch -> retry
                    for slot, raw_text, reasoning_text in zip(
                        slots, completions, _reasoning_for(completions)
                    ):
                        parsed = _parse_output(
                            raw_text, output_schema, reasoning=reasoning_text
                        )
                        if parsed is None:
                            unfilled.append(slot)
                        else:
                            results[slot] = parsed
                    if unfilled:
                        next_pending.append((prompt, unfilled))
        pending = next_pending

    if pending:
        failed = sum(len(slots) for _, slots in pending)
        log.warning(
            "%d sample(s) failed to parse after %d retries; "
            "returning None for those slots",
            failed,
            max_retries,
        )
    return results


def default_extract_answer(text: str) -> str | None:
    r"""Extract a final-answer token from a free-form response.

    Tries explicit answer markers in order -- ``ANSWER: <n>``, ``\boxed{<n>}``,
    ``#### <n>``, ``final answer ... <n>`` -- and falls back to the last number
    in the text. Returns the answer as a normalized string (thousands separators
    stripped; other commas, as in ``1,2``, kept), or ``None`` when no number is
    found.

    This is the default segmentation used by :func:`self_taught_reasoner`; pass a
    custom ``extract_answer`` for non-numeric answers or different markers.
    """
    if not text:
        return None
    for pattern in _ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            # A trailing comma is punctuation ("1000, so ..."), not part of it.
            return _normalize_answer(matches[-1].rstrip(","))
    return None


def _normalize_answer(value: Any) -> str:
    """Stripped string form, with thousands separators dropped from numbers."""
    text = str(value).strip()
    return text.replace(",", "") if _GROUPED_NUMBER_RE.fullmatch(text) else text


def _normalized_equal(a: Any, b: Any) -> bool:
    """Compare two answers by normalized string form (handles int/str mismatch).

    Both sides get the same normalization, so a reference of ``"1,000"`` matches
    the ``"1000"`` that :func:`default_extract_answer` returns. Other commas are
    kept, so a list such as ``"1,2"`` or a decimal comma such as ``"3,5"`` does
    not match ``"12"`` or ``"35"``.
    """
    return _normalize_answer(a) == _normalize_answer(b)


@op(name="alter.self_taught_reasoner")
def self_taught_reasoner(
    items: list[dict],
    prompt_fn: PromptTemplate,
    model: Model,
    *,
    n: int = 8,
    reference_field: str = "final_answer",
    extract_answer: Callable[[str], Any] | None = None,
    match_fn: Callable[[Any, Any], bool] | None = None,
    filter_fn: Callable[[list[dict]], list[dict]] | None = None,
    response_field: str = "response",
    thinking_field: str = "thinking",
    answer_field: str = "answer",
    student_answer_field: str = "student_final_answer",
    split_thinking: bool = True,
    temperature: float = 1.0,
    max_tokens: int | None = None,
    batch_size: int = 10,
) -> list[dict]:
    r"""Self-Taught Reasoner (STaR): keep a student model's own correct solutions.

    This implements the Self-Taught Reasoner (STaR) approach of Zelikman et al.,
    "STaR: Bootstrapping Reasoning With Reasoning" (https://arxiv.org/abs/2203.14465):
    the model generates its own reasoning traces and only those that reach the
    verified answer are retained as training targets.

    For each input item, ``n`` responses are sampled from ``model`` (the
    "student"). A response is only kept when the final answer matches the item's
    reference, so the surviving rows carry student-generated reasoning traces that
    arrive at a verified answer.

    Acceptance is decided in one of two ways:

    * **Answer matching (default):**  A response is kept when
      ``match_fn(student_answer, item[reference_field])`` is true (default:
      string-normalized equality).
    * **Custom filter:** Pass ``filter_fn`` -- any ``list[dict] -> list[dict]``
      callable.

    Each kept response becomes a separate output row (e.g. if ``n`` responses
    are sampled and two are correct, there will be two new output rows created).
    An item with zero matching responses contributes no rows. The reasoning is
    the provider's separate reasoning field (``GenerateResult.reasoning``) when
    the model returns one, and otherwise the inline ``<think>`` block split out
    via `pyforger.acquire.split_reasoning`.

    Args:
        items: Input rows. Each must contain ``reference_field``.
        prompt_fn: Format string (keys filled from the item) or callable
            ``(dict) -> Prompt`` producing the student prompt for each item.
        model: Student model to sample from.
        n: Number of responses to sample per item.
        reference_field: Key holding each item's ground-truth answer.
        extract_answer: ``str -> answer`` extractor that fills
            ``student_answer_field`` on every candidate. It still runs when
            ``filter_fn`` is given (a warning is logged), and the filter sees
            its output.
        match_fn: ``(student_answer, reference) -> bool`` for the default path.
            Ignored when ``filter_fn`` is given.
        filter_fn: Optional ``list[dict] -> list[dict]`` acceptance filter over
            candidate rows. Overrides the default answer-matching path.
        response_field: Output key for the raw student response.
        thinking_field: Output key for the split-out reasoning trace.
        answer_field: Output key for the answer portion of the response.
        student_answer_field: Key holding the extracted student answer on each
            candidate row (also visible to ``filter_fn``).
        split_thinking: When True, fill thinking/answer from the API reasoning,
            or by splitting an inline ``<think>`` block when there is none.
            When False, the whole response is stored in ``answer_field`` and
            ``thinking_field`` is left empty.
        temperature: Sampling temperature passed to the model.
        max_tokens: Max tokens per response (model default when None).
        batch_size: Number of items per ``model.generate_batch()`` call.

    Returns:
        Output rows, in item order, one per accepted response.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not items:
        return []

    log = _get_logger()
    if extract_answer is not None and filter_fn is not None:
        log.warning(
            "self_taught_reasoner: extract_answer still runs when filter_fn is "
            "given; it fills %r on the candidate rows filter_fn receives.",
            student_answer_field,
        )
    extract = extract_answer if extract_answer is not None else default_extract_answer
    match = match_fn if match_fn is not None else _normalized_equal

    prompts: list[Prompt] = [_resolve_prompt(prompt_fn, item) for item in items]

    # Sample n responses per item, batching items together for throughput.
    item_responses: list[tuple[dict, list[str]]] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        batch_results = model.generate_batch(
            batch, n=n, temperature=temperature, max_tokens=max_tokens
        )
        if len(batch_results) != len(batch):
            log.warning(
                "generate_batch returned %d results for %d prompts",
                len(batch_results),
                len(batch),
            )
        # zip drops items left without results; they contribute no rows.
        item_responses.extend(zip(items[start : start + batch_size], batch_results))

    # Build one candidate row per generated response.
    candidates: list[dict] = []
    for item, responses in item_responses:
        # Providers that return reasoning separately (GenerateResult.reasoning)
        # leave it out of the content; otherwise it is inline as <think>...</think>.
        reasoning = getattr(responses, "reasoning", None) or []
        for i, raw in enumerate(responses):
            api_thinking = reasoning[i] if i < len(reasoning) else None
            if not split_thinking:
                thinking, answer = "", raw
            elif api_thinking:
                thinking, answer = api_thinking.strip(), raw.strip()
            else:
                thinking, answer = split_reasoning(raw)
            candidate = {
                **item,
                response_field: raw,
                thinking_field: thinking,
                answer_field: answer,
                student_answer_field: extract(raw),
            }
            candidates.append(candidate)

    # Decide which candidates to keep.
    if filter_fn is not None:
        kept = filter_fn(candidates)
    else:
        kept = [
            candidate
            for candidate in candidates
            if candidate[student_answer_field] is not None
            and match(
                candidate[student_answer_field],
                get_field(candidate, reference_field, "alter.self_taught_reasoner"),
            )
        ]

    # The student-generated answer becomes the ground truth for kept rows.
    for candidate in kept:
        student_answer = candidate.get(student_answer_field)
        if student_answer is not None:
            candidate[reference_field] = student_answer

    log.info(
        "self_taught_reasoner: kept %d/%d responses across %d items",
        len(kept),
        len(candidates),
        len(items),
    )
    return kept
