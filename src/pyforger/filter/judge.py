"""LLM-as-a-judge filters.

``by_llm_score``: each item is rated by N independent judge calls; kept iff at
least ``agreement_threshold`` of judges score it ≥ ``score_threshold``.

``by_comparative_judge``: rows sharing a ``group_by`` value are candidates for
the same input; a judge ranks them side by side and the top ``keep`` survive.
"""
from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Hashable
from typing import Any, Union

from pyforger import _tracking
from pyforger.acquire.model import Model, Prompt, _cache_kwargs
from pyforger.op import op
from pyforger.utils import get_field, get_formatted_logger

logger = get_formatted_logger("Judge")

# A format string over the row's fields plus the judge placeholders (e.g.
# "{input}", "{output}"), or a callable from that dict to a prompt.
JudgePromptTemplate = Union[str, Callable[[dict], Prompt]]


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
    temperature: float = 0.0,
    prompt_template: JudgePromptTemplate | None = None,
    score_field: str | None = None,
    explanation_field: str | None = None,
) -> list[dict]:
    """Keep items where ``num_judges`` LLM judges sufficiently agree on quality.

    All ``len(items) * num_judges`` judge calls go out in one
    ``model.generate_batch`` call (one ``n=1`` request per judge), so they run
    concurrently. Judge responses are cached separately from the thresholds, so
    re-running with a different ``score_threshold`` or ``agreement_threshold``
    makes no new calls. ``_no_cache=True`` also re-asks the judges.

    ``prompt_template`` replaces the default prompt. A string is filled with
    ``str.format_map`` over the row's fields plus ``{input}`` (the
    ``input_field`` value), ``{output}``, ``{criteria}``, and ``{max_score}``,
    which win over row fields of the same name; write literal braces as ``{{``
    and ``}}``. A callable receives that same dict and returns a prompt. If the
    prompt does not contain ``"score"`` (the JSON key, in double quotes), the
    instruction to answer with ``{"score": ..., "explanation": ...}`` is
    appended to it.

    Only the text after a closed ``</think>`` block is read; a response with an
    unclosed ``<think>`` is unparseable. The score comes from the last JSON
    object with a ``score`` key, also inside code fences or surrounding text,
    else from the last ``"score": <number>`` pattern. Both paths read a string
    score's leading number (``"4/5"`` is 4) and keep fractions (``4.5`` stays
    ``4.5``); the score must lie in ``[1, max_score]``. Unparseable responses
    count as a judge that did not approve.

    With ``score_field`` set, kept rows carry the mean of the judges' parsed
    scores; with ``explanation_field`` set, a list of their explanations.

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

    prompts = [
        _score_prompt(
            item, input_field, output_field, criteria, max_score, prompt_template
        )
        for item in items
    ]
    responses = _judge_responses(
        [prompt for prompt in prompts for _ in range(num_judges)],
        model=model,
        temperature=temperature,
    )
    kept: list[dict] = []
    # One judge: its score. Several: the fraction of judges approving, which is
    # what agreement_threshold is compared with. None when nothing parsed.
    row_scores: list[float | int | None] = []
    parse_failures = 0
    for i, item in enumerate(items):
        verdicts = [
            _parse_score_response(raw, max_score)
            for raw in responses[i * num_judges : (i + 1) * num_judges]
        ]
        parsed = [score for score, _ in verdicts if score is not None]
        parse_failures += num_judges - len(parsed)
        approvals = sum(1 for score in parsed if score >= score_threshold)
        if approvals / num_judges >= agreement_threshold:
            extra: dict[str, Any] = {}
            if score_field is not None:
                extra[score_field] = sum(parsed) / len(parsed)
            if explanation_field is not None:
                extra[explanation_field] = [
                    why for score, why in verdicts if score is not None and why
                ]
            kept.append({**item, **extra} if extra else item)
        if not parsed:
            row_scores.append(None)
        elif num_judges == 1:
            row_scores.append(parsed[0])
        else:
            row_scores.append(approvals / num_judges)
    _warn_unparsed("by_llm_score", parse_failures)
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
    return _generate_all(prompts, model, temperature)


@op(name="filter.by_comparative_judge")
def by_comparative_judge(
    items: list[dict],
    *,
    group_by: str,
    model: Model,
    input_field: str,
    output_field: str,
    criteria: str = "overall quality, relevance, and correctness",
    keep: int = 1,
    num_judges: int = 1,
    prompt_template: JudgePromptTemplate | None = None,
    seed: int = 0,
    temperature: float = 0.0,
    rank_field: str | None = None,
    explanation_field: str | None = None,
) -> list[dict]:
    """Keep the ``keep`` best rows of each group, as ranked side by side by a judge.

    Rows with the same ``group_by`` value are candidates for the same input,
    e.g. four sampled answers to one question. One prompt shows the input (from
    the group's first row) once and the candidates' ``output_field`` values
    labelled A, B, C, ...; the judge answers with JSON
    ``{"ranking": ["C", "A", "B"], "explanation": "..."}``, best first; entries
    may carry words after the label (``"A (best)"``), but an entry naming no
    shown label makes that judge's answer unparseable. Text before a closed
    ``</think>`` is ignored, and an unclosed ``<think>`` is unparseable. The
    candidates are shuffled per group (seeded by ``seed`` and the group), and
    judge ``j`` sees that order rotated by ``j``, so each candidate takes each
    position equally often to spread position bias. ``num_judges > 1`` needs
    ``temperature > 0``. Rankings combine by
    Borda count: in a group of ``m`` the candidate ranked ``p``-th (from 0)
    gets ``m - 1 - p`` points, and candidates a judge leaves out share the
    remaining points. Ties go to the row that came first. Every group of two or
    more rows is judged, whatever ``keep`` is; a single row is kept without a
    call. A group none of whose judges answered parseably is dropped. Kept rows
    stay in input order.

    All calls go out in one ``model.generate_batch`` call (``n=1`` each). Judge
    responses are cached apart from ``keep``, so a new ``keep`` makes no new
    calls.

    ``prompt_template`` replaces the default prompt. A string is filled with
    ``str.format_map`` over the group's first row plus ``{input}``,
    ``{candidates}`` (the labelled outputs), ``{labels}`` (e.g. ``"A, B, C"``),
    ``{num_candidates}``, and ``{criteria}``; a callable receives that dict. If
    the prompt does not contain ``"ranking"``, the JSON answer instruction is
    appended.

    With ``rank_field`` set, kept rows carry their rank in the group (1 = best);
    with ``explanation_field`` set, the list of the group's judge explanations
    (empty for a single row).

    In a ``@step`` report the per-row score is the row's rank in its group
    (1 = best), kept when at most ``keep``.
    """
    if keep < 1:
        raise ValueError("keep must be at least 1")
    if num_judges < 1:
        raise ValueError("num_judges must be at least 1")
    if num_judges > 1 and temperature == 0.0:
        raise ValueError(
            "temperature must be > 0.0 when num_judges > 1; with more judges "
            "than candidates, judges see the same order and would answer alike"
        )

    groups: dict[Hashable, list[int]] = {}
    for i, item in enumerate(items):
        value = get_field(item, group_by, "by_comparative_judge")
        groups.setdefault(_hashable(value), []).append(i)
    # Judged independently of keep, so the calls (and cache) never depend on it.
    judged = [(key, idx) for key, idx in groups.items() if len(idx) > 1]

    # orders[g][j][k] is the candidate (index into group g) labelled k for judge j.
    orders: list[list[list[int]]] = []
    prompts: list[Prompt] = []
    for key, idx in judged:
        group = [items[i] for i in idx]
        group_orders = []
        base = list(range(len(group)))
        random.Random(f"{seed}|{key!r}").shuffle(base)
        for j in range(num_judges):
            # Judge j sees the shuffled list rotated by j, so over len(group)
            # judges every candidate is shown once in every position.
            shift = j % len(base)
            order = base[shift:] + base[:shift]
            group_orders.append(order)
            prompts.append(
                _ranking_prompt(
                    group,
                    order,
                    input_field,
                    output_field,
                    criteria,
                    prompt_template,
                )
            )
        orders.append(group_orders)
    responses = _ranking_responses(prompts, model=model, temperature=temperature)

    # Rank within the group, 1 = best; None when no judge answered parseably.
    ranks: list[int | None] = [None] * len(items)
    explanations: list[list[str]] = [[] for _ in items]
    for idx in groups.values():
        if len(idx) == 1:
            ranks[idx[0]] = 1
    parse_failures = 0
    for g, (_, idx) in enumerate(judged):
        points = [0.0] * len(idx)
        whys: list[str] = []
        any_parsed = False
        for j, order in enumerate(orders[g]):
            raw = responses[g * num_judges + j]
            ranking = _parse_ranking_response(raw, order)
            if ranking is None:
                parse_failures += 1
                continue
            any_parsed = True
            why = _ranking_explanation(raw)
            if why:
                whys.append(why)
            for candidate, pts in _borda_points(ranking, len(idx)).items():
                points[candidate] += pts
        if not any_parsed:
            continue
        best_first = sorted(range(len(idx)), key=lambda c: (-points[c], c))
        for r, candidate in enumerate(best_first, start=1):
            ranks[idx[candidate]] = r
            explanations[idx[candidate]] = whys

    _warn_unparsed("by_comparative_judge", parse_failures)
    _tracking.record_scores("rank in group", ranks, keep="le", threshold=float(keep))
    kept: list[dict] = []
    for item, rank, reasons in zip(items, ranks, explanations):
        if rank is None or rank > keep:
            continue
        extra: dict[str, Any] = {}
        if rank_field is not None:
            extra[rank_field] = rank
        if explanation_field is not None:
            extra[explanation_field] = list(reasons)
        kept.append({**item, **extra} if extra else item)
    return kept


@op(name="filter.by_comparative_judge.rankings")
def _ranking_responses(
    prompts: list[Prompt], *, model: Model, temperature: float
) -> list[str]:
    """Return one raw ranking response per prompt (one prompt per group and judge)."""
    return _generate_all(prompts, model, temperature)


def _generate_all(prompts: list[Prompt], model: Model, temperature: float) -> list[str]:
    if not prompts:
        return []
    # One request per prompt with n=1: several providers reject n>1. Each prompt
    # is built from one row (or group), so it keys that row in the request
    # cache: changing one row re-asks only its judges, and the op scope keeps
    # other judge ops apart. Repeated prompts (one per judge) get new slots.
    results = model.generate_batch(
        prompts,
        n=1,
        temperature=temperature,
        **_cache_kwargs(model.generate_batch, cache_rows=list(prompts)),
    )
    return [result[0] if result else "" for result in results]


def _warn_unparsed(op_name: str, count: int) -> None:
    if count:
        logger.warning("%s: %d judge responses unparseable", op_name, count)


def _hashable(value: Any) -> Hashable:
    """Return ``value`` as a dict key; e.g. a list becomes its JSON text."""
    if isinstance(value, Hashable):
        return value
    return json.dumps(value, sort_keys=True, default=str)


# --------------------------------------------------------------------- prompts


def _score_prompt(
    item: dict,
    input_field: str,
    output_field: str,
    criteria: str,
    max_score: int,
    template: JudgePromptTemplate | None,
) -> Prompt:
    input_text = get_field(item, input_field, "by_llm_score")
    output_text = get_field(item, output_field, "by_llm_score")
    if template is None:
        return (
            f"Rate the quality of the following output on a scale of 1 to "
            f"{max_score}.\n"
            f"Criteria: {criteria}\n\n"
            f"Input: {input_text}\n\n"
            f"Output: {output_text}\n\n"
            f'Respond with JSON: {{"score": <integer>, "explanation": "<text>"}}'
        )
    context = {
        **item,
        "input": input_text,
        "output": output_text,
        "criteria": criteria,
        "max_score": max_score,
    }
    instruction = (
        f'Respond with JSON: {{"score": <integer from 1 to {max_score}>, '
        f'"explanation": "<text>"}}'
    )
    return _with_instruction(
        _fill(template, context), '"score"', instruction, "by_llm_score"
    )


def _ranking_prompt(
    group: list[dict],
    order: list[int],
    input_field: str,
    output_field: str,
    criteria: str,
    template: JudgePromptTemplate | None,
) -> Prompt:
    first = group[0]
    input_text = get_field(first, input_field, "by_comparative_judge")
    labels = [_label(k) for k in range(len(order))]
    candidates = "\n\n".join(
        f"Candidate {label}:\n"
        f"{get_field(group[c], output_field, 'by_comparative_judge')}"
        for label, c in zip(labels, order)
    )
    instruction = (
        'Respond with JSON: {"ranking": [<every candidate label, best first>], '
        '"explanation": "<text>"}'
    )
    if template is None:
        return (
            f"Compare these {len(order)} candidate outputs for the same input "
            f"and rank them from best to worst.\n"
            f"Criteria: {criteria}\n\n"
            f"Input: {input_text}\n\n"
            f"{candidates}\n\n"
            f"{instruction}"
        )
    context = {
        **first,
        "input": input_text,
        "candidates": candidates,
        "labels": ", ".join(labels),
        "num_candidates": len(order),
        "criteria": criteria,
    }
    return _with_instruction(
        _fill(template, context), '"ranking"', instruction, "by_comparative_judge"
    )


def _fill(template: JudgePromptTemplate, context: dict) -> Prompt:
    if isinstance(template, str):
        return template.format_map(context)
    return template(context)


def _with_instruction(
    prompt: Prompt, key: str, instruction: str, caller: str
) -> Prompt:
    """Append ``instruction`` to ``prompt`` unless it already mentions ``key``."""
    if isinstance(prompt, str):
        return prompt if key in prompt else f"{prompt}\n\n{instruction}"
    if not isinstance(prompt, list) or not prompt:
        raise TypeError(
            f"{caller}: prompt_template must give a string or a non-empty list "
            f"of messages, got {type(prompt).__name__}"
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
_EXPLANATION_RE = re.compile(r"[\"']explanation[\"']\s*:\s*\"((?:[^\"\\]|\\.)*)\"")
# No closing bracket needed, so a truncated '"ranking": ["C", "A"' still parses.
_RANKING_RE = re.compile(r"[\"']ranking[\"']\s*:\s*\[([^\]]*)")
# The label at the start of an entry: "B", "(b)", "Candidate A:", "A (best)".
_LABEL_RE = re.compile(r"^\W*(?:candidate\s+)?([A-Z]{1,2})\b", re.I)
# A leading number: "4", "4.5", "4/5", "4 out of 5".
_NUMBER_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)")


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


def _parse_score_response(
    raw: str, max_score: int
) -> tuple[int | float | None, str | None]:
    """Return ``(score, explanation)``; score is None when missing or off-scale."""
    text = _answer_text(raw)
    if text is None:
        return None, None
    parsed = _find_json_with(text, "score")
    if parsed is not None:
        why = parsed.get("explanation")
        return (
            _to_score(parsed["score"], max_score),
            why if isinstance(why, str) else None,
        )
    matches = _SCORE_RE.findall(text)
    if not matches:
        return None, None
    why_match = _EXPLANATION_RE.search(text)
    return _to_score(matches[-1], max_score), why_match.group(1) if why_match else None


def _parse_ranking_response(raw: str, order: list[int]) -> list[int] | None:
    """Map a judge's ranking of labels to candidate indices, best first.

    Entries may carry words after the label ("A (best)", "Candidate B:").
    Repeated labels are skipped. Any entry that names no shown label makes the
    whole ranking unparseable (None), so a garbled entry can never push the
    judge's top pick to the bottom. ``order[k]`` is the candidate shown with
    label k.
    """
    text = _answer_text(raw)
    if text is None:
        return None
    parsed = _find_json_with(text, "ranking")
    if parsed is not None:
        ranking = parsed["ranking"]
        if isinstance(ranking, str):
            parts = [p for p in re.split(r"\s*[,>;\n]\s*", ranking) if p]
            ranking = ranking.split() if len(parts) == 1 else parts
        if not isinstance(ranking, list):
            return None
    else:
        match = _RANKING_RE.search(text)
        if match is None:
            return None
        ranking = [entry for entry in match.group(1).split(",") if entry.strip()]
    by_label = {_label(k): c for k, c in enumerate(order)}
    candidates: list[int] = []
    for entry in ranking:
        label_match = _LABEL_RE.match(str(entry))
        candidate = by_label.get(label_match.group(1).upper()) if label_match else None
        if candidate is None:
            return None
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates or None


def _ranking_explanation(raw: str) -> str | None:
    """Return the ``explanation`` of a ranking response, if it has one."""
    text = _answer_text(raw)
    if text is None:
        return None
    parsed = _find_json_with(text, "ranking")
    if parsed is not None:
        why = parsed.get("explanation")
        return why if isinstance(why, str) else None
    match = _EXPLANATION_RE.search(text)
    return match.group(1) if match else None


def _borda_points(ranking: list[int], m: int) -> dict[int, float]:
    """Borda points per candidate: ``m - 1 - p`` for position ``p`` (from 0).

    Candidates missing from a partial ranking share the points left over, e.g.
    ranking ``[2]`` of 3 gives ``{2: 2, 0: 0.5, 1: 0.5}``.
    """
    points = {c: float(m - 1 - p) for p, c in enumerate(ranking)}
    rest = [c for c in range(m) if c not in points]
    if rest:
        share = sum(range(m - len(ranking))) / len(rest)
        points.update({c: share for c in rest})
    return points
