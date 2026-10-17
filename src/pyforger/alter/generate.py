"""Building block for conditional LM generation (conditioned on upstream input)."""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any, Union

from pyforger import _tracking
from pyforger.acquire._model_utils import (
    _accepts_kwarg,
    _build_prompt,
    _is_transient,
    _parse_output,
    _reasoning_for,
    _report_provider_failures,
)
from pyforger.acquire.model import Model, Prompt, _cache_kwargs, split_reasoning
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


@op(name="alter.generate_from_context", samples_arg="samples_per_item")
def generate_from_context(
    items: list[dict],
    prompt_fn: PromptTemplate,
    model: Model,
    *,
    samples_per_item: int = 1,
    output_schema: dict[str, Any] | None = None,
    temperature: float | None = None,
    max_retries: int = 3,
    batch_size: int | None = None,
    merge: bool = False,
    output_field: str | None = None,
) -> list[dict | None] | list[dict]:
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
        temperature: Sampling temperature passed to the model; None (default)
            uses the model's own default (1.0 for ``Model``, Qwen3's 0.6 for
            ``TinkerModel``).
        max_retries: Additional attempts for prompts whose output fails to parse
            or whose request fails at the provider (e.g. a 504).
        batch_size: Number of prompts per ``model.generate_batch()`` call. None
            (default) sends all prompts in one call; the model's
            ``max_concurrency`` bounds how many requests run at once.
        merge: Return the input rows with the output attached instead of the
            bare outputs: ``{**item, **parsed}`` per output (the output wins
            when keys clash; a warning names them), plus ``sample_idx`` when
            ``samples_per_item > 1``. Failed slots are dropped.
        output_field: With ``merge=True``, nest the output under this key
            instead, e.g. ``"solution"`` gives ``{**item, "solution": parsed}``.

    Returns:
        By default, a list of length ``len(items) * samples_per_item`` in input
        order, grouped by item (item ``i``'s samples occupy slots
        ``i * samples_per_item`` to ``(i + 1) * samples_per_item``). Each slot
        holds the parsed output dict, or ``None`` if the prompt failed to parse
        or failed at the provider after ``max_retries``; when any slot failed at
        the provider, the result is not cached, so a rerun asks again (the
        request cache replays the completions that arrived). With
        ``merge=True``, the merged rows of the slots that succeeded. When the
        model returns a reasoning trace separately in `GenerateResult.reasoning`,
        it is included under the `thinking_trace` key. Inline `<think>` traces (e.g.
        from `LocalModel`, which supplies `None` for that metadata) are not extracted;
        use `split_reasoning` to separate them out as needed.
    """
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if samples_per_item <= 0:
        raise ValueError("samples_per_item must be positive")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if output_field is not None and not merge:
        raise ValueError("output_field needs merge=True")

    log = _get_logger()

    prompts: list[Prompt] = [
        _build_prompt(_resolve_prompt(prompt_fn, item), output_schema) for item in items
    ]

    results: list[dict | None] = [None] * (len(prompts) * samples_per_item)

    # A transient provider error (e.g. a 504) comes back in the prompt's slot
    # instead of failing the whole batch; models whose generate_batch lacks the
    # keyword raise as before.
    catch: dict[str, Any] = (
        {"return_exceptions": True}
        if _accepts_kwarg(getattr(model, "generate_batch", None), "return_exceptions")
        else {}
    )

    # Each pending entry is (item index, slots still needed). Retries re-ask
    # only the missing slots, up to max_retries more attempts.
    pending: list[tuple[int, list[int]]] = [
        (i, list(range(i * samples_per_item, (i + 1) * samples_per_item)))
        for i in range(len(prompts))
    ]
    # Item index -> type name of its latest provider failure, e.g.
    # "InternalServerError"; cleared when a later attempt gets a response.
    provider_errors: dict[int, str] = {}

    for attempt in range(1 + max_retries):
        if not pending:
            break
        next_pending: list[tuple[int, list[int]]] = []

        # generate_batch uses one n per call, so group prompts needing the same
        # number of completions (on the first pass all need samples_per_item).
        by_count: dict[int, list[tuple[int, list[int]]]] = {}
        for entry in pending:
            by_count.setdefault(len(entry[1]), []).append(entry)

        for n, group in by_count.items():
            chunk = batch_size or len(group)
            for start in range(0, len(group), chunk):
                batch = group[start : start + chunk]
                # Each prompt is keyed on its row, so editing one row re-asks
                # only that row. A retry (after a parse failure or a provider
                # error) is salted, e.g. "retry 1, n=2", and gets new
                # completions, not the same failed ones.
                batch_results = model.generate_batch(
                    [prompts[i] for i, _ in batch],
                    temperature=temperature,
                    n=n,
                    **_cache_kwargs(
                        model.generate_batch,
                        cache_rows=[items[i] for i, _ in batch],
                        cache_salt=f"retry {attempt}, n={n}" if attempt else None,
                    ),
                    **catch,
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
                for (i, slots), completions in zip(batch, batch_results):
                    if isinstance(completions, BaseException):
                        # A bad key, unknown model, or 400 fails every retry.
                        if not _is_transient(completions):
                            raise completions
                        provider_errors[i] = type(completions).__name__
                        next_pending.append((i, slots))
                        continue
                    provider_errors.pop(i, None)
                    unfilled = slots[len(completions) :]
                    if unfilled and catch:
                        # Fewer completions than asked: some of the prompt's
                        # requests failed (return_exceptions keeps the rest).
                        provider_errors[i] = "missing completions"
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
                        next_pending.append((i, unfilled))
        pending = next_pending

    failed_parse = [slots for i, slots in pending if i not in provider_errors]
    if failed_parse:
        log.warning(
            "%d sample(s) failed to parse after %d retries; "
            "returning None for those slots",
            sum(len(slots) for slots in failed_parse),
            max_retries,
        )
    failed_provider = [provider_errors[i] for i, _ in pending if i in provider_errors]
    if failed_provider:
        _report_provider_failures(
            f"alter.generate_from_context: {len(failed_provider)} of {len(prompts)} "
            f"prompts failed after {max_retries} "
            f"{'retry' if max_retries == 1 else 'retries'}; their slots are None",
            failed_provider,
        )
    if not merge:
        return results
    return _merge_outputs(items, results, samples_per_item, output_field, log)


def _merge_outputs(
    items: list[dict],
    outputs: list[dict | None],
    samples_per_item: int,
    output_field: str | None,
    log: logging.Logger,
) -> list[dict]:
    """Attach each parsed output to its input row; drop failed slots.

    ``{**item, **parsed}``, or ``{**item, output_field: parsed}``, plus
    ``sample_idx`` (0, 1, ...) when there are several samples per item.
    """
    merged: list[dict] = []
    clashes: set[str] = set()
    for slot, parsed in enumerate(outputs):
        if parsed is None:
            continue
        item = items[slot // samples_per_item]
        row: dict[str, Any]
        if output_field is not None:
            clashes.update({output_field} & item.keys())
            row = {**item, output_field: parsed}
        else:
            clashes.update(parsed.keys() & item.keys())
            row = {**item, **parsed}
        if samples_per_item > 1:
            # e.g. a sample_idx left by an earlier merge=True step.
            clashes.update({"sample_idx"} & row.keys())
            row["sample_idx"] = slot % samples_per_item
        merged.append(row)
    if clashes:
        message = (
            f"alter.generate_from_context: merge=True replaces the input's "
            f"{sorted(clashes)}; pass another output_field= (or rename the input "
            "keys) to keep both"
        )
        log.warning(message)
        _tracking.add_op_warning(message)
    return merged


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


@op(name="alter.self_taught_reasoner", samples_arg="n")
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
    temperature: float | None = None,
    max_tokens: int | None = None,
    batch_size: int | None = None,
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
        temperature: Sampling temperature passed to the model; None (default)
            uses the model's own default (1.0 for ``Model``, Qwen3's 0.6 for
            ``TinkerModel``).
        max_tokens: Max tokens per response (model default when None).
        batch_size: Number of items per ``model.generate_batch()`` call. None
            (default) sends all items in one call.

    Returns:
        Output rows, in item order, one per accepted response.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if batch_size is not None and batch_size <= 0:
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
    chunk = batch_size or len(prompts)
    for start in range(0, len(prompts), chunk):
        batch = prompts[start : start + chunk]
        batch_results = model.generate_batch(
            batch,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            **_cache_kwargs(
                model.generate_batch, cache_rows=items[start : start + chunk]
            ),
        )
        if len(batch_results) != len(batch):
            log.warning(
                "generate_batch returned %d results for %d prompts",
                len(batch_results),
                len(batch),
            )
        # zip drops items left without results; they contribute no rows.
        item_responses.extend(zip(items[start : start + chunk], batch_results))

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
