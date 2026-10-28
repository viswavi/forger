"""Building block for conditional LM generation (conditioned on upstream input)."""
from __future__ import annotations

import contextvars
import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Union

from pyforger import _tracking
from pyforger.acquire._model_utils import (
    _accepts_kwarg,
    _build_prompt,
    _failure_label,
    _is_row_error,
    _is_transient,
    _parse_output,
    _raise_if_systemic,
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

# Reasoning that opens by working out the reply format, e.g. "We need answer
# JSON only with solution_code? User says 'Respond with your reasoning' but...".
_FORMAT_TALK_RE = re.compile(
    r"\bjson\b|required keys|output format|response format", re.I
)
_FORMAT_TALK_CHARS = 400  # how far into a trace to look
_FORMAT_TALK_MIN_TRACES = 5
_FORMAT_TALK_SHARE = 0.2


def _get_logger() -> logging.Logger:
    """Lazy logger to avoid circular import with utils."""
    from pyforger.utils import get_formatted_logger as _gf

    return _gf(__name__)


def _with_hint(prompt: Prompt, hint: str) -> Prompt:
    """``prompt`` with ``hint`` after its text (a chat's last user message)."""
    if isinstance(prompt, str):
        return f"{prompt}\n\n{hint}"
    messages = [dict(m) for m in prompt]
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            message["content"] = f"{message['content']}\n\n{hint}"
            return messages  # type: ignore[return-value]
    return [*messages, {"role": "user", "content": hint}]  # type: ignore[list-item]


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
    retry_truncated: bool = False,
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
            JSON-parsed and validated; parse failures are retried. When None,
            each response is stored as ``{"text": ...}``. Only the keys are checked:
            a response needs every key, keys outside the schema are dropped, and
            the values' types are shown to the model but not checked. A ```json
            fence or prose around the object is tolerated.
        temperature: Sampling temperature passed to the model; None (default)
            uses the model's own default (1.0 for ``Model``, Qwen3's 0.6 for
            ``TinkerModel``).
        max_retries: Additional attempts for prompts whose output fails to
            parse (a reply arrived but did not parse). A request that fails at
            the provider (e.g. a 504) is retried by ``model`` itself
            (``Model(num_retries=)``) and not asked again here. A sample that
            fails to parse because it stopped at the model's max_tokens is not
            retried unless ``retry_truncated=True``: at the same max_tokens a
            retry usually stops there again. The step report counts such samples.
        batch_size: Number of prompts per ``model.generate_batch()`` call. None
            (default) sends all prompts in one call; the model's
            ``max_concurrency`` bounds how many requests run at once.
        merge: Return the input rows with the output attached instead of the
            bare outputs: ``{**item, **parsed}`` per output (the output wins
            when keys clash; a warning names them), plus ``sample_idx`` when
            ``samples_per_item > 1``. Failed slots are dropped.
        output_field: With ``merge=True``, nest the output under this key
            instead, e.g. ``"solution"`` gives ``{**item, "solution": parsed}``.
        retry_truncated: Also retry samples that failed to parse after stopping
            at max_tokens (see ``max_retries``).

    Returns:
        By default, a list of length ``len(items) * samples_per_item`` in input
        order, grouped by item (item ``i``'s samples occupy slots
        ``i * samples_per_item`` to ``(i + 1) * samples_per_item``). Each slot
        holds the parsed output dict, or ``None`` if the prompt failed to parse
        after ``max_retries`` or failed at the provider after the model's own
        retries. A row error (the provider's content filter blocked it, or
        the prompt is longer than the context window) also gives ``None``,
        unless every prompt (at least 2) hit one: then the first is raised.
        When any slot failed at the provider, the result is not
        cached, so a rerun asks again (the request cache replays the
        completions that arrived). With
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
    # only the slots that failed to parse, up to max_retries more attempts.
    pending: list[tuple[int, list[int]]] = [
        (i, list(range(i * samples_per_item, (i + 1) * samples_per_item)))
        for i in range(len(prompts))
    ]
    # Item index -> label of its latest provider failure, e.g.
    # "InternalServerError" or "content filter". Model already retried the
    # request (num_retries) if a retry could help, so the op does not ask
    # again: those slots stay None.
    provider_errors: dict[int, str] = {}
    row_errors: dict[int, BaseException] = {}
    # Slots whose output failed to parse after stopping at max_tokens.
    cut_off: set[int] = set()
    # Slots kept although they stopped at max_tokens (e.g. output_schema=None
    # keeps any text), so their text is cut off.
    cut_kept: set[int] = set()
    # Slot -> the start of its reasoning when it talks about the reply format,
    # else "" (slots without reasoning are left out). Only with output_schema.
    format_talk: dict[int, str] = {}

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
                # only that row. A retry (after a parse failure) is salted,
                # e.g. "retry 1, n=2", and gets new completions, not the same
                # unparseable ones.
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
                        # A content-filter block or an over-long prompt fails
                        # only this row; a bad key, unknown model, or other
                        # 400 fails every retry.
                        if _is_row_error(completions):
                            row_errors[i] = completions
                        elif not _is_transient(completions):
                            raise completions
                        provider_errors[i] = _failure_label(completions)
                        continue
                    unfilled: list[int] = []
                    if len(completions) < len(slots) and catch:
                        # Fewer completions than asked: some of the prompt's
                        # requests failed (return_exceptions keeps the rest).
                        provider_errors[i] = "missing completions"
                    elif len(completions) < len(slots):
                        unfilled = slots[len(completions) :]
                    reasons = getattr(completions, "finish_reasons", None) or []
                    for k, (slot, raw_text, reasoning_text) in enumerate(
                        zip(slots, completions, _reasoning_for(completions))
                    ):
                        parsed = _parse_output(
                            raw_text, output_schema, reasoning=reasoning_text
                        )
                        stopped = k < len(reasons) and reasons[k] == "length"
                        if output_schema is not None:
                            talk = _format_talk(raw_text, reasoning_text)
                            if talk is not None:
                                format_talk[slot] = talk
                        if parsed is not None:
                            results[slot] = parsed
                            cut_off.discard(slot)
                            if stopped:
                                cut_kept.add(slot)
                            else:
                                cut_kept.discard(slot)
                        elif stopped:
                            cut_off.add(slot)
                            if retry_truncated:
                                unfilled.append(slot)
                        else:
                            unfilled.append(slot)
                    if unfilled:
                        next_pending.append((i, unfilled))
        pending = next_pending

    # With merge=True a failed slot's row is dropped; otherwise its slot is None.
    def lost(n: int) -> str:
        if merge:
            return f"{n} row{'' if n == 1 else 's'} dropped"
        return f"{n} output{'' if n == 1 else 's'} None"

    def kept(n: int) -> str:
        noun = "row" if merge else "output"
        return f"{n} {noun}{'' if n == 1 else 's'} kept with cut-off text"

    if cut_off:
        log.warning(
            "%d sample(s) stopped at max_tokens before any parseable output; %s"
            "returning None for those slots",
            len(cut_off),
            "" if retry_truncated else "not retried (retry_truncated=False); ",
        )
    if cut_off or cut_kept:
        # Leads the op's max_tokens warning: "20 rows dropped (...), because ...".
        outcome = []
        if cut_off:
            outcome.append(lost(len(cut_off)))
        if cut_kept:
            outcome.append(kept(len(cut_kept)))
        _tracking.add_truncation_outcome(", ".join(outcome))
    failed_parse = [
        slot for _, slots in pending for slot in slots if slot not in cut_off
    ]
    if failed_parse:
        n = len(failed_parse)
        message = (
            f"alter.generate_from_context: {lost(n)}: {n} sample"
            f"{'' if n == 1 else 's'} did not parse after {max_retries} retries"
        )
        log.warning(message)
        _tracking.add_op_warning(message, "parse_failure")
    _report_format_talk(format_talk, log)
    _raise_if_systemic(list(row_errors.values()), len(prompts))
    if provider_errors:
        _report_provider_failures(
            f"alter.generate_from_context: {len(provider_errors)} of {len(prompts)} "
            "prompts failed after the model's retries; their slots are None",
            list(provider_errors.values()),
        )
    if not merge:
        return results
    return _merge_outputs(items, results, samples_per_item, output_field, log)


def _format_talk(raw_text: str, reasoning_text: str | None) -> str | None:
    """The start of a completion's reasoning if it talks about the reply format.

    Returns "" for reasoning that does not, and None for a completion without
    reasoning (neither ``reasoning_text`` nor an inline ``<think>`` block).
    """
    if reasoning_text is None:
        head = raw_text.lstrip()
        if not head.startswith("<think>"):
            return None
        reasoning_text = head[len("<think>") :]
    start = reasoning_text[:_FORMAT_TALK_CHARS]
    return " ".join(start.split()) if _FORMAT_TALK_RE.search(start) else ""


def _report_format_talk(format_talk: dict[int, str], log: logging.Logger) -> None:
    """Warn when many reasoning traces open by working out the reply format.

    e.g. "We need answer JSON only? User says 'Respond with your reasoning,
    then a code block' but ...": the prompt asks for one format and
    output_schema's instruction for another. Such reasoning is a poor training
    target, so the warning says so.
    """
    talking = [t for t in format_talk.values() if t]
    if len(format_talk) < _FORMAT_TALK_MIN_TRACES or not talking:
        return
    if len(talking) / len(format_talk) < _FORMAT_TALK_SHARE:
        return
    example = talking[0][:80] + ("…" if len(talking[0]) > 80 else "")
    message = (
        f"alter.generate_from_context: {len(talking)} of {len(format_talk)} "
        f'reasoning traces open by discussing the output format (e.g. "{example}") '
        "→ the prompt's own format instructions likely conflict with "
        "output_schema's; if this reasoning becomes a training target, generate "
        "without output_schema and parse the answer yourself"
    )
    log.warning(message)
    _tracking.add_op_warning(message, "format_talk")


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
        _tracking.add_op_warning(message, "merge_clash")
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
    verifier: Callable[[str, dict], bool | None] | None = None,
    max_workers: int | None = None,
    rationalize: bool = True,
    hint_fn: PromptTemplate | None = None,
    rationalized_field: str = "rationalized",
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
    * **Verifier:** Pass ``verifier`` -- a ``(answer, row) -> bool | None``
      callable that sees the whole row, e.g.
      ``filter.program_verifier("tests")`` to run each answer's code on the
      row's tests. A response is kept when it returns True. Calls run
      ``max_workers`` at a time.
    * **Custom filter:** Pass ``filter_fn`` -- any ``list[dict] -> list[dict]``
      callable.

    Each kept response becomes a separate output row (e.g. if ``n`` responses
    are sampled and two are correct, there will be two new output rows created).
    An item with zero matching responses contributes no rows. The reasoning is
    the provider's separate reasoning field (``GenerateResult.reasoning``) when
    the model returns one, and otherwise the inline ``<think>`` block split out
    via `pyforger.acquire.split_reasoning`.

    **Rationalization** (the paper's second pass, on by default): every item
    left with no accepted response is sampled ``n`` more times from a prompt
    that also gives a hint, and the accepted responses are kept as if the
    student had answered the unhinted prompt: rows carry the item's fields,
    not the hinted prompt. This gives rows for items the student never solves
    on its own, in the student's own words. The default hint, as in the paper,
    is the item's reference (``prompt_fn``'s prompt plus "Hint: a correct
    answer is: <item[reference_field]>"); items without one are not retried.
    ``hint_fn`` gives another hint (e.g. only the final code, or why the task
    cannot be done). As in the paper, responses that mention the hint are not
    filtered out. Every row gets ``rationalized_field`` (False for the first
    pass, True for the second). ``rationalize=False`` turns this off; with
    ``filter_fn`` it is off, since the filter judges all candidates together
    and cannot say which items went unsolved.

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
        verifier: Optional ``(answer, row) -> bool | None`` acceptance check,
            given ``answer_field`` (the response after its reasoning) and the
            candidate row (the item's fields plus the response fields). With
            it, ``reference_field`` is not needed and is left unchanged.
        max_workers: Verifier calls at a time (default 8).
        rationalize: Re-sample unsolved items with a hint (default True).
        hint_fn: Format string or ``(dict) -> Prompt`` for the hinted prompt;
            None (default) appends the item's ``reference_field`` as the hint.
            Needs answer matching or ``verifier``, not ``filter_fn``.
        rationalized_field: Output key marking rationalized rows (set when
            rationalization runs).
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
    if filter_fn is not None and verifier is not None:
        raise ValueError("pass filter_fn or verifier, not both")
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if extract_answer is not None and filter_fn is not None:
        log.warning(
            "self_taught_reasoner: extract_answer still runs when filter_fn is "
            "given; it fills %r on the candidate rows filter_fn receives.",
            student_answer_field,
        )
    extract = extract_answer if extract_answer is not None else default_extract_answer
    match = match_fn if match_fn is not None else _normalized_equal

    if hint_fn is not None and filter_fn is not None:
        raise ValueError(
            "hint_fn needs answer matching or verifier: filter_fn judges all "
            "candidates together, so the unsolved items are unknown"
        )
    if hint_fn is not None and not rationalize:
        raise ValueError("hint_fn is only used with rationalize=True")
    rationalizing = rationalize and filter_fn is None

    def default_hint(item: dict) -> Prompt:
        """``prompt_fn``'s prompt, with the item's reference as a hint."""
        hint = f"Hint: a correct answer is:\n{item[reference_field]}"
        return _with_hint(_resolve_prompt(prompt_fn, item), hint)

    def sample(template: PromptTemplate, batch_items: list[dict]) -> list[list[dict]]:
        """``n`` candidate rows per item, from ``template``'s prompts."""
        prompts = [_resolve_prompt(template, item) for item in batch_items]
        per_item: list[list[dict]] = [[] for _ in batch_items]
        # Sample n responses per item, batching items together for throughput.
        chunk = batch_size or len(prompts)
        for start in range(0, len(prompts), chunk):
            batch = prompts[start : start + chunk]
            batch_results = model.generate_batch(
                batch,
                n=n,
                temperature=temperature,
                max_tokens=max_tokens,
                **_cache_kwargs(
                    model.generate_batch, cache_rows=batch_items[start : start + chunk]
                ),
            )
            if len(batch_results) != len(batch):
                log.warning(
                    "generate_batch returned %d results for %d prompts",
                    len(batch_results),
                    len(batch),
                )
            # Items left without results contribute no rows.
            for offset, responses in enumerate(batch_results[: len(batch)]):
                per_item[start + offset] = candidates_for(
                    batch_items[start + offset], responses
                )
        return per_item

    def candidates_for(item: dict, responses: list[str]) -> list[dict]:
        """One candidate row per generated response."""
        # Providers that return reasoning separately (GenerateResult.reasoning)
        # leave it out of the content; otherwise it is inline as <think>...</think>.
        reasoning = getattr(responses, "reasoning", None) or []
        rows = []
        for i, raw in enumerate(responses):
            api_thinking = reasoning[i] if i < len(reasoning) else None
            if not split_thinking:
                thinking, answer = "", raw
            elif api_thinking:
                thinking, answer = api_thinking.strip(), raw.strip()
            else:
                thinking, answer = split_reasoning(raw)
            rows.append(
                {
                    **item,
                    response_field: raw,
                    thinking_field: thinking,
                    answer_field: answer,
                    student_answer_field: extract(raw),
                }
            )
        return rows

    def accept(candidates: list[dict]) -> list[dict]:
        """The candidates to keep, in order."""
        if filter_fn is not None:
            return filter_fn(candidates)
        if verifier is not None:
            return _verified(candidates, verifier, answer_field, max_workers or 8)
        return [
            candidate
            for candidate in candidates
            if candidate[student_answer_field] is not None
            and match(
                candidate[student_answer_field],
                get_field(candidate, reference_field, "alter.self_taught_reasoner"),
            )
        ]

    def accept_per_item(per_item: list[list[dict]]) -> list[list[dict]]:
        """`accept` over all candidates at once, regrouped by item."""
        kept_ids = {id(c) for c in accept([c for cs in per_item for c in cs])}
        return [[c for c in cs if id(c) in kept_ids] for cs in per_item]

    first = sample(prompt_fn, items)
    n_candidates = sum(len(cs) for cs in first)
    if not rationalizing:
        kept = accept([c for cs in first for c in cs])
    else:
        kept_per_item = accept_per_item(first)
        # The default hint is the reference, so items without one are not retried.
        unsolved = [
            i
            for i, cs in enumerate(kept_per_item)
            if not cs
            and (hint_fn is not None or items[i].get(reference_field) is not None)
        ]
        hint_template = hint_fn if hint_fn is not None else default_hint
        hinted = sample(hint_template, [items[i] for i in unsolved]) if unsolved else []
        n_candidates += sum(len(cs) for cs in hinted)
        for cs in kept_per_item:
            for candidate in cs:
                candidate[rationalized_field] = False
        for i, cs in zip(unsolved, accept_per_item(hinted)):
            for candidate in cs:
                candidate[rationalized_field] = True
            kept_per_item[i] = cs
        kept = [c for cs in kept_per_item for c in cs]
        log.info(
            "self_taught_reasoner: rationalization kept %d responses for %d of %d "
            "unsolved items",
            sum(len(kept_per_item[i]) for i in unsolved),
            sum(1 for i in unsolved if kept_per_item[i]),
            len(unsolved),
        )

    # The student-generated answer becomes the ground truth for kept rows.
    for candidate in kept if verifier is None else []:
        student_answer = candidate.get(student_answer_field)
        if student_answer is not None:
            candidate[reference_field] = student_answer

    log.info(
        "self_taught_reasoner: kept %d/%d responses across %d items",
        len(kept),
        n_candidates,
        len(items),
    )
    return kept


def _context_run() -> Callable[..., Any]:
    """A copy of this context's run, typed for Executor.submit (as in model.py)."""
    run: Callable[..., Any] = contextvars.copy_context().run
    return run


def _verified(
    candidates: list[dict],
    verifier: Callable[[str, dict], bool | None],
    answer_field: str,
    workers: int,
) -> list[dict]:
    """The candidates ``verifier`` accepts, in order; ``workers`` calls at a time."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _context_run(),
                verifier,
                candidate[answer_field],
                candidate,
            )
            for candidate in candidates
        ]
        try:
            verdicts = [f.result() for f in futures]
        except BaseException:
            for f in futures:
                f.cancel()
            raise
    return [c for c, ok in zip(candidates, verdicts) if ok is True]
