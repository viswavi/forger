"""LLM-as-a-judge filter.

``by_llm_score``: each row is scored by N independent judge calls against a
rubric that describes every score level; the judge writes feedback (the
output's mistakes) before the score. A row is kept iff at least
``agreement_threshold`` of judges score it ≥ ``score_threshold``. With
``group_by``, the candidates for one input are scored together in one prompt.
"""
from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Hashable
from typing import Any, Union

from pyforger import _tracking
from pyforger.acquire.model import Model, Prompt
from pyforger.op import op
from pyforger.utils import get_field, get_formatted_logger

logger = get_formatted_logger("Judge")

# A format string over the row's fields plus the judge placeholders (e.g.
# "{input}", "{output}", "{rubric}"), or a callable from that dict to a prompt.
JudgePromptTemplate = Union[str, Callable[[dict], Prompt]]

# One description per level of the default 1-5 scale. Each level names how bad
# the output's worst mistakes are, so the score follows from the feedback.
DEFAULT_RUBRIC: dict[int, str] = {
    1: "Fails the task: wrong, off-topic, a refusal, or unusable. Its mistakes "
    "against the criteria are fundamental.",
    2: "Attempts the task but has major mistakes or omissions against the "
    "criteria; most of it would need to be redone.",
    3: "Meets the core of the criteria but has clear mistakes or gaps that a "
    "careful reviewer would have to fix.",
    4: "Meets the criteria with only minor shortcomings (e.g. a small omission "
    "or imprecise wording) that do not make it wrong or misleading.",
    5: "Fully meets the criteria with no mistakes or shortcomings; a careful "
    "expert would change nothing.",
}


@op(name="filter.by_llm_score")
def by_llm_score(
    items: list[dict],
    *,
    model: Model,
    input_field: str,
    output_field: str,
    score_threshold: int = 3,
    max_score: int = 5,
    num_judges: int = 1,
    agreement_threshold: float = 1.0,
    criteria: str = "overall quality, relevance, and correctness",
    rubric: dict[int, str] | None = None,
    group_by: str | None = None,
    temperature: float = 0.0,
    prompt_template: JudgePromptTemplate | None = None,
    score_field: str | None = None,
    explanation_field: str | None = None,
) -> list[dict]:
    """Keep items where ``num_judges`` LLM judges sufficiently agree on quality.

    The default prompt shows ``criteria`` and a rubric with a description for
    every score level, and asks the judge to write feedback first (every
    mistake or shortcoming of the output, or that there are none), then give
    the rubric level that feedback matches, as JSON
    ``{"feedback": "...", "score": 3}``. ``rubric`` replaces the level
    descriptions and must cover exactly ``1..max_score``; the default,
    ``DEFAULT_RUBRIC``, covers the default ``max_score=5``, so any other
    ``max_score`` needs a ``rubric`` (unless a ``prompt_template`` does not use
    ``{rubric}``).

    With ``group_by``, rows sharing that field's value are candidates for the
    same input (e.g. four sampled answers to one question) and are judged in
    one prompt: the input (from the group's first row) once, then the
    candidates labelled A, B, C, ...; the judge answers
    ``{"candidates": [{"label": "A", "feedback": "...", "score": 4}, ...]}``,
    scoring each on the same rubric. The candidates are shuffled per group and
    judge ``j`` sees that order rotated by ``j``, so each candidate takes each
    position equally often. A group of one row gets the single-output prompt.
    Each candidate is then kept or dropped by the same threshold rules as
    without ``group_by``; a missing or unparseable candidate score is no vote
    for that candidate only.

    All judge calls go out in one ``model.generate_batch`` call (one ``n=1``
    request per prompt and judge), so they run concurrently. Judge responses
    are cached separately from the thresholds, so re-running with a different
    ``score_threshold`` or ``agreement_threshold`` makes no new calls.
    ``_no_cache=True`` also re-asks the judges. ``num_judges > 1`` needs
    ``temperature > 0``.

    ``prompt_template`` replaces the default prompt. A string is filled with
    ``str.format_map`` over the row's fields plus ``{input}`` (the
    ``input_field`` value), ``{output}``, ``{criteria}``, ``{max_score}``, and
    ``{rubric}`` (the rendered rubric), which win over row fields of the same
    name; write literal braces as ``{{`` and ``}}``. A group's prompt also gets
    ``{candidates}`` (the labelled outputs, also given as ``{output}``),
    ``{labels}`` (e.g. ``"A, B, C"``), and ``{num_candidates}``. A callable
    receives that same dict and returns a prompt. If the prompt does not
    contain the JSON key in double quotes (``"score"``, or ``"candidates"`` for
    a group), the feedback-then-score answer instruction is appended to it.

    Only the text after a closed ``</think>`` block is read; a response with an
    unclosed ``<think>`` is unparseable. The score comes from the last JSON
    object with a ``score`` key, also inside code fences or surrounding text,
    else from the last ``"score": <number>`` pattern; the feedback from its
    ``feedback`` key, or ``explanation`` for older templates. Both paths read a
    string score's leading number (``"4/5"`` is 4) and keep fractions (``4.5``
    stays ``4.5``); the score must lie in ``[1, max_score]``. Unparseable
    responses count as a judge that did not approve.

    Returns:
        The kept rows, in input order. With ``score_field`` set, each carries
        the mean of its judges' parsed scores; with ``explanation_field`` set,
        the list of their feedback. Otherwise rows are unchanged.

    In a ``@step`` report the per-row score is the judge's score when
    ``num_judges == 1``, and otherwise the fraction of judges scoring at least
    ``score_threshold`` (compared with ``agreement_threshold``).
    """
    if num_judges < 1:
        raise ValueError("num_judges must be at least 1")
    if not (0.0 < agreement_threshold <= 1.0):
        raise ValueError("agreement_threshold must be in (0, 1]")
    if score_threshold > max_score:
        raise ValueError(
            f"score_threshold ({score_threshold}) must not exceed "
            f"max_score ({max_score})"
        )
    if num_judges > 1 and temperature == 0.0:
        raise ValueError(
            "temperature must be > 0.0 when num_judges > 1; "
            "deterministic sampling defeats multi-judge consensus"
        )
    rubric_text = _rubric_text(rubric, max_score)

    # Row indices by group value; without group_by every row is alone.
    by_value: dict[Hashable, list[int]] = {}
    for i, item in enumerate(items):
        value = i if group_by is None else get_field(item, group_by, "by_llm_score")
        by_value.setdefault(_hashable(value), []).append(i)
    groups = list(by_value.values())

    # orders[g][j][k] is the candidate (index into group g) labelled k for judge j.
    orders: list[list[list[int]]] = []
    prompts: list[Prompt] = []
    for key, idx in by_value.items():
        group = [items[i] for i in idx]
        if len(idx) == 1:
            prompt = _score_prompt(
                group[0],
                input_field,
                output_field,
                criteria,
                max_score,
                rubric_text,
                prompt_template,
            )
            orders.append([[0]] * num_judges)
            prompts.extend([prompt] * num_judges)
            continue
        base = list(range(len(idx)))
        random.Random(repr(key)).shuffle(base)
        group_orders = []
        for j in range(num_judges):
            # Judge j sees the shuffled list rotated by j, so over len(group)
            # judges every candidate is shown once in every position.
            shift = j % len(base)
            order = base[shift:] + base[:shift]
            group_orders.append(order)
            prompts.append(
                _group_prompt(
                    group,
                    order,
                    input_field,
                    output_field,
                    criteria,
                    max_score,
                    rubric_text,
                    prompt_template,
                )
            )
        orders.append(group_orders)
    responses = _judge_responses(prompts, model=model, temperature=temperature)

    # verdicts[i][j] is judge j's (score, feedback) for row i.
    verdicts: list[list[Verdict]] = [[] for _ in items]
    position = 0
    for idx, group_orders in zip(groups, orders):
        for order in group_orders:
            raw = responses[position]
            position += 1
            if len(idx) == 1:
                verdicts[idx[0]].append(_parse_score_response(raw, max_score))
                continue
            for c, verdict in enumerate(_parse_group_response(raw, order, max_score)):
                verdicts[idx[c]].append(verdict)

    kept: list[dict] = []
    # One judge: its score. Several: the fraction of judges approving, which is
    # what agreement_threshold is compared with. None when nothing parsed.
    row_scores: list[float | int | None] = []
    parse_failures = 0
    for item, row_verdicts in zip(items, verdicts):
        parsed = [score for score, _ in row_verdicts if score is not None]
        parse_failures += num_judges - len(parsed)
        approvals = sum(1 for score in parsed if score >= score_threshold)
        if approvals / num_judges >= agreement_threshold:
            extra: dict[str, Any] = {}
            if score_field is not None:
                extra[score_field] = sum(parsed) / len(parsed)
            if explanation_field is not None:
                extra[explanation_field] = [
                    why for score, why in row_verdicts if score is not None and why
                ]
            kept.append({**item, **extra} if extra else item)
        if not parsed:
            row_scores.append(None)
        elif num_judges == 1:
            row_scores.append(parsed[0])
        else:
            row_scores.append(approvals / num_judges)
    _warn_unparsed(parse_failures, len(items) * num_judges)
    if num_judges == 1:
        _tracking.record_scores(
            "score", row_scores, keep="ge", threshold=float(score_threshold)
        )
    else:
        _tracking.record_scores(
            f"approval >={score_threshold}",
            row_scores,
            keep="ge",
            threshold=agreement_threshold,
        )
    return kept


@op(name="filter.by_llm_score.judge")
def _judge_responses(
    prompts: list[Prompt], *, model: Model, temperature: float
) -> list[str]:
    """Return one raw judge response per prompt (prompts repeat per judge)."""
    if not prompts:
        return []
    # One request per prompt with n=1: several providers reject n>1.
    results = model.generate_batch(prompts, n=1, temperature=temperature)
    return [result[0] if result else "" for result in results]


def _warn_unparsed(count: int, total: int) -> None:
    if count:
        logger.warning("by_llm_score: %d of %d judge scores unparseable", count, total)


def _hashable(value: Any) -> Hashable:
    """Return ``value`` as a dict key; e.g. a list becomes its JSON text."""
    if isinstance(value, Hashable):
        return value
    return json.dumps(value, sort_keys=True, default=str)


# --------------------------------------------------------------------- prompts


def _rubric_text(rubric: dict[int, str] | None, max_score: int) -> str | None:
    """Render the rubric as ``Score k: ...`` lines; None if there is none.

    There is no default for a ``max_score`` other than 5; a prompt that needs
    the rubric then raises (see ``_fill``).
    """
    if rubric is None:
        if max_score != len(DEFAULT_RUBRIC):
            return None
        rubric = DEFAULT_RUBRIC
    else:
        levels = set(range(1, max_score + 1))
        keys = set(rubric)
        if keys != levels or any(isinstance(k, bool) for k in keys):
            missing = sorted(levels - keys)
            extra = sorted(keys - levels, key=repr)
            raise ValueError(
                f"by_llm_score: rubric must describe exactly the levels 1 to "
                f"{max_score}; missing {missing}, unexpected {extra}"
            )
        empty = [k for k in sorted(rubric) if not str(rubric[k]).strip()]
        if empty:
            raise ValueError(f"by_llm_score: rubric levels {empty} are empty")
    return "\n".join(f"Score {k}: {rubric[k]}" for k in range(1, max_score + 1))


def _score_instruction(max_score: int) -> str:
    return (
        "First write feedback: list every mistake or shortcoming of the output "
        "against the criteria, or say that there are none. Then give the score "
        "whose rubric description matches the output; the score must follow "
        "from your feedback and the rubric.\n"
        'Respond with JSON, feedback before score: {"feedback": "<text>", '
        f'"score": <integer from 1 to {max_score}>}}'
    )


def _group_instruction(labels: list[str], max_score: int) -> str:
    return (
        "For each candidate, first write feedback: list every mistake or "
        "shortcoming of that candidate against the criteria, or say that there "
        "are none. Then give the score whose rubric description matches it; "
        "each score must follow from that candidate's feedback and the rubric. "
        "Score each candidate on its own against the rubric, not by rank: "
        "several candidates may share a score.\n"
        f"Respond with JSON, one entry per candidate ({', '.join(labels)}), "
        'feedback before score: {"candidates": [{"label": "A", "feedback": '
        f'"<text>", "score": <integer from 1 to {max_score}>}}, ...]}}'
    )


def _score_prompt(
    item: dict,
    input_field: str,
    output_field: str,
    criteria: str,
    max_score: int,
    rubric_text: str | None,
    template: JudgePromptTemplate | None,
) -> Prompt:
    input_text = get_field(item, input_field, "by_llm_score")
    output_text = get_field(item, output_field, "by_llm_score")
    instruction = _score_instruction(max_score)
    if template is None:
        return (
            f"Score the output below for the given input on a scale of 1 to "
            f"{max_score}, using the criteria and the score rubric.\n\n"
            f"Criteria: {criteria}\n\n"
            f"Score rubric:\n{_require_rubric(rubric_text, max_score)}\n\n"
            f"Input: {input_text}\n\n"
            f"Output: {output_text}\n\n"
            f"{instruction}"
        )
    context = {
        **item,
        "input": input_text,
        "output": output_text,
        "criteria": criteria,
        "max_score": max_score,
    }
    if rubric_text is not None:
        context["rubric"] = rubric_text
    return _with_instruction(
        _fill(template, context, max_score), '"score"', instruction
    )


def _group_prompt(
    group: list[dict],
    order: list[int],
    input_field: str,
    output_field: str,
    criteria: str,
    max_score: int,
    rubric_text: str | None,
    template: JudgePromptTemplate | None,
) -> Prompt:
    first = group[0]
    input_text = get_field(first, input_field, "by_llm_score")
    labels = [_label(k) for k in range(len(order))]
    candidates = "\n\n".join(
        f"Candidate {label}:\n{get_field(group[c], output_field, 'by_llm_score')}"
        for label, c in zip(labels, order)
    )
    instruction = _group_instruction(labels, max_score)
    if template is None:
        return (
            f"Score each of the {len(order)} candidate outputs below for the "
            f"same input on a scale of 1 to {max_score}, using the criteria and "
            f"the score rubric.\n\n"
            f"Criteria: {criteria}\n\n"
            f"Score rubric:\n{_require_rubric(rubric_text, max_score)}\n\n"
            f"Input: {input_text}\n\n"
            f"{candidates}\n\n"
            f"{instruction}"
        )
    context = {
        **first,
        "input": input_text,
        "output": candidates,
        "candidates": candidates,
        "labels": ", ".join(labels),
        "num_candidates": len(order),
        "criteria": criteria,
        "max_score": max_score,
    }
    if rubric_text is not None:
        context["rubric"] = rubric_text
    return _with_instruction(
        _fill(template, context, max_score), '"candidates"', instruction
    )


def _no_rubric_error(max_score: int) -> ValueError:
    return ValueError(
        f"by_llm_score: there is no default rubric for max_score={max_score} "
        f"(DEFAULT_RUBRIC covers 1 to {len(DEFAULT_RUBRIC)}); pass "
        f"rubric={{1: '...', ..., {max_score}: '...'}}"
    )


def _require_rubric(rubric_text: str | None, max_score: int) -> str:
    if rubric_text is None:
        raise _no_rubric_error(max_score)
    return rubric_text


def _fill(template: JudgePromptTemplate, context: dict, max_score: int) -> Prompt:
    try:
        if isinstance(template, str):
            return template.format_map(context)
        return template(context)
    except KeyError as error:
        # "{rubric}" with no rubric for this max_score.
        if error.args == ("rubric",) and "rubric" not in context:
            raise _no_rubric_error(max_score) from None
        raise


def _with_instruction(prompt: Prompt, key: str, instruction: str) -> Prompt:
    """Append ``instruction`` to ``prompt`` unless it already mentions ``key``."""
    if isinstance(prompt, str):
        return prompt if key in prompt else f"{prompt}\n\n{instruction}"
    if not isinstance(prompt, list) or not prompt:
        raise TypeError(
            "by_llm_score: prompt_template must give a string or a non-empty "
            f"list of messages, got {type(prompt).__name__}"
        )
    if any(key in str(message.get("content", "")) for message in prompt):
        return prompt
    last = prompt[-1]
    if last.get("role") == "user":
        content = f"{last.get('content', '')}\n\n{instruction}"
        return [*prompt[:-1], {**last, "content": content}]
    return [*prompt, {"role": "user", "content": instruction}]


def _label(k: int) -> str:
    """Candidate label: A..Z, then AA, AB, ..."""
    label = ""
    k += 1
    while k:
        k, rem = divmod(k - 1, 26)
        label = chr(ord("A") + rem) + label
    return label


# --------------------------------------------------------------------- parsing

_SCORE_RE = re.compile(r"[\"']score[\"']\s*:\s*\"?(-?\d+(?:\.\d+)?)")
_FEEDBACK_RE = re.compile(
    r"[\"'](?:feedback|explanation)[\"']\s*:\s*\"((?:[^\"\\]|\\.)*)\""
)
_CANDIDATE_LABEL_RE = re.compile(r"[\"']label[\"']\s*:\s*[\"']([^\"']*)[\"']")
# The label at the start of an entry: "B", "(b)", "Candidate A:", "A (best)".
_LABEL_RE = re.compile(r"^\W*(?:candidate\s+)?([A-Z]{1,2})\b", re.I)
# A leading number: "4", "4.5", "4/5", "4 out of 5".
_NUMBER_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)")

Verdict = tuple[Union[int, float, None], Union[str, None]]


def _answer_text(raw: Any) -> str | None:
    """Return the text after a closed ``</think>`` block; None if unclosed.

    A draft inside the block must not count, e.g.
    ``<think>maybe {"score": 2}</think>{"score": 5}`` is read as ``{"score": 5}``.
    """
    if not isinstance(raw, str):
        return None
    if "</think>" in raw:
        return raw.rsplit("</think>", 1)[1]
    if "<think>" in raw:
        return None
    return raw


def _find_json_with(raw: str, key: str) -> dict | None:
    """Return the last top-level JSON object in ``raw`` that has ``key``.

    Handles bare JSON, code fences, and JSON inside prose, e.g.
    ``Sure! ```json {"score": 4} ``` ``. An object without ``key`` is searched
    inside, so ``{"result": {"score": 4}}`` gives ``{"score": 4}``. The last
    object wins, like the regex fallback's last match.
    """
    decoder = json.JSONDecoder()
    found = None
    start = raw.find("{")
    while start != -1:
        try:
            obj, end = decoder.raw_decode(raw, start)
        except json.JSONDecodeError:
            obj, end = None, start + 1
        if isinstance(obj, dict) and key in obj:
            found = obj
        else:
            end = start + 1  # look for the key in nested objects
        start = raw.find("{", end)
    return found


def _to_score(value: Any, max_score: int) -> int | float | None:
    """Read a judge score: a number, or a string's leading number ("4/5" is 4).

    Fractions are kept (``4.5`` stays ``4.5``); None when off ``[1, max_score]``.
    """
    if isinstance(value, str):
        match = _NUMBER_RE.match(value)
        if match is None:
            return None
        text = match.group(1)
        value = float(text) if "." in text else int(text)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not 1 <= value <= max_score:
        return None
    return value


def _feedback(entry: dict) -> str | None:
    """The ``feedback`` of a parsed answer, else its legacy ``explanation``."""
    for key in ("feedback", "explanation"):
        value = entry.get(key)
        if isinstance(value, str):
            return value
    return None


def _parse_score_response(raw: str, max_score: int) -> Verdict:
    """Return ``(score, feedback)``; score is None when missing or off-scale."""
    text = _answer_text(raw)
    if text is None:
        return None, None
    parsed = _find_json_with(text, "score")
    if parsed is not None:
        return _to_score(parsed["score"], max_score), _feedback(parsed)
    matches = _SCORE_RE.findall(text)
    if not matches:
        return None, None
    why_match = _FEEDBACK_RE.search(text)
    return _to_score(matches[-1], max_score), why_match.group(1) if why_match else None


def _candidate_entries(text: str) -> list[tuple[Any, Any, str | None]]:
    """Return ``(label, raw score, feedback)`` per candidate entry of an answer.

    Reads ``{"candidates": [{"label": "A", "feedback": ..., "score": 4}]}`` or
    ``{"candidates": {"A": {...}}}``; without such an object (e.g. a truncated
    answer), each ``"label": "A"`` pattern up to the next one.
    """
    parsed = _find_json_with(text, "candidates")
    if parsed is not None:
        candidates = parsed["candidates"]
        if isinstance(candidates, dict):
            pairs = list(candidates.items())
        elif isinstance(candidates, list):
            pairs = [
                (entry.get("label", entry.get("candidate")), entry)
                for entry in candidates
                if isinstance(entry, dict)
            ]
        else:
            return []
        return [
            (label, entry.get("score"), _feedback(entry))
            if isinstance(entry, dict)
            else (label, entry, None)
            for label, entry in pairs
        ]
    matches = list(_CANDIDATE_LABEL_RE.finditer(text))
    entries: list[tuple[Any, Any, str | None]] = []
    for k, match in enumerate(matches):
        end = matches[k + 1].start() if k + 1 < len(matches) else len(text)
        segment = text[match.end() : end]
        score = _SCORE_RE.search(segment)
        why = _FEEDBACK_RE.search(segment)
        entries.append(
            (
                match.group(1),
                score.group(1) if score else None,
                why.group(1) if why else None,
            )
        )
    return entries


def _parse_group_response(raw: str, order: list[int], max_score: int) -> list[Verdict]:
    """Return one ``(score, feedback)`` per candidate of a group, by index.

    ``order[k]`` is the candidate shown with label k. A candidate the answer
    leaves out, or whose score does not parse, gets ``(None, None)``; the other
    candidates keep their scores. Entries naming no shown label are skipped,
    and a repeated label keeps its first parsed score.
    """
    verdicts: list[Verdict] = [(None, None)] * len(order)
    text = _answer_text(raw)
    if text is None:
        return verdicts
    by_label = {_label(k): c for k, c in enumerate(order)}
    for label, value, why in _candidate_entries(text):
        label_match = _LABEL_RE.match(str(label))
        candidate = by_label.get(label_match.group(1).upper()) if label_match else None
        if candidate is None or verdicts[candidate][0] is not None:
            continue
        score = _to_score(value, max_score)
        if score is not None:
            verdicts[candidate] = (score, why)
    return verdicts
