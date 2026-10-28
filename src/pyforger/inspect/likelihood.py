"""How likely the student model finds each row's target, against its own responses."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field
from rich.console import Group, RenderableType
from rich.padding import Padding
from rich.text import Text

from pyforger._render import (
    WIDTH,
    _clean,
    _console,
    _emit,
    _fit_table,
    _plain,
    _rule,
    fmt_num,
)
from pyforger.acquire.model import TargetLogprobs
from pyforger.op import op, tracked
from pyforger.utils import get_field

_LN2 = math.log(2)


class RowLikelihood(BaseModel):
    """One response scored by the student: its length and how surprising it is.

    ``bits`` is ``-log2`` of the response's probability given its prompt, the
    sum over its tokens. ``excess_bits`` is what it costs beyond the student's
    own rate (bits per token on its own responses) for the same number of
    tokens; about 0 for a response the student would write itself.
    """

    row: int
    tokens: int
    bits: float
    reasoning_tokens: int = 0
    reasoning_bits: float = 0.0
    truncated: bool = False
    excess_bits: float = 0.0

    @property
    def bits_per_token(self) -> float | None:
        """Mean bits per token, or None for an empty response."""
        return self.bits / self.tokens if self.tokens else None


class LikelihoodReport(BaseModel):
    """What ``student_log_likelihood`` measured.

    ``rows`` has one entry per row given, in order. ``own`` scores the
    student's own response to an evenly spaced subset of the prompts (``row``
    is the prompt's row); ``own_bits_per_token`` is their median rate and the
    zero point of every ``excess_bits``.
    """

    model_id: str
    target_field: str
    rows: list[RowLikelihood] = Field(default_factory=list)
    own: list[RowLikelihood] = Field(default_factory=list)
    own_bits_per_token: float | None = None
    own_hit_max_tokens: int = 0

    @property
    def total_bits(self) -> float:
        """Bits summed over every target."""
        return sum(r.bits for r in self.rows)

    @property
    def total_excess_bits(self) -> float:
        """Excess bits summed over every target."""
        return sum(r.excess_bits for r in self.rows)


def _row(i: int, score: TargetLogprobs) -> RowLikelihood:
    bits = [-lp / _LN2 for lp in score.logprobs]
    return RowLikelihood(
        row=i,
        tokens=len(bits),
        bits=sum(bits),
        reasoning_tokens=score.reasoning_tokens,
        reasoning_bits=sum(bits[: score.reasoning_tokens]),
        truncated=score.truncated,
    )


def _evenly_spaced(n: int, k: int) -> list[int]:
    """``k`` indices spread over ``range(n)``, e.g. n=10, k=3 -> [0, 3, 6]."""
    if k >= n:
        return list(range(n))
    return [i * n // k for i in range(k)]


@op(name="inspect.student_log_likelihood.targets")
def _target_scores(
    rows: list[dict], *, model: Any, prompt_field: str, target_field: str
) -> list[TargetLogprobs]:
    caller = "student_log_likelihood"
    prompts = [get_field(r, prompt_field, caller) for r in rows]
    targets = [str(get_field(r, target_field, caller)) for r in rows]
    return model.target_logprobs(prompts, targets)


@op(name="inspect.student_log_likelihood.own")
def _own_scores(
    prompts: list[Any], *, model: Any, max_tokens: int | None
) -> tuple[list[TargetLogprobs], int]:
    """Sample one response per prompt and score it; also count hits of max_tokens."""
    results = model.generate_batch(prompts, n=1, max_tokens=max_tokens)
    hit_max = sum(
        1
        for r in results
        if (getattr(r, "finish_reasons", None) or [None])[0] == "length"
    )
    return model.target_logprobs(prompts, [r[0] for r in results]), hit_max


@tracked("inspect.student_log_likelihood")
def student_log_likelihood(
    rows: list[dict],
    model: Any,
    *,
    prompt_field: str = "prompt",
    target_field: str = "response",
    own_samples: int = 200,
    max_tokens: int | None = None,
    top_k: int = 5,
) -> LikelihoodReport:
    """Print and return how likely the student finds each row's target, in bits.

    Scores every target (e.g. an SFT ``"<think>...</think>answer"``) under the
    student, as SFT trains on it: bits = ``-log2`` of each token's probability
    given the prompt and the tokens before it. A bare log-likelihood has no
    natural scale, so the report sets it beside the student's own responses to
    ``own_samples`` of the same prompts, sampled at the model's temperature:

    - bits per token, targets against the student's own: how foreign the text
      is, token by token;
    - tokens per response, targets against the student's own;
    - excess bits: each target's bits minus what the student's own rate would
      cost for that many tokens. About 0 for responses the student would write
      itself (e.g. STaR traces); the sum over the dataset grows with both how
      foreign and how long the targets are, i.e. roughly how far SFT on them
      must move the student. It is a description, not a validated predictor of
      what SFT will do.

    Reasoning (through ``</think>``) and answer are reported apart. Rows are not
    modified. Needs a model with ``target_logprobs``, e.g. ``LocalModel``; the
    scores and the sampled responses are cached.

    Args:
        rows: The rows (dicts).
        model: The student, e.g. ``LocalModel("Qwen/Qwen3-4B")``.
        prompt_field: Field holding the prompt (string or chat messages).
        target_field: Field holding the response to score.
        own_samples: How many prompts (evenly spaced) the student answers itself
            for the baseline; 0 skips sampling and reports no excess bits.
        max_tokens: Cap on the student's own responses (model default if None).
        top_k: Rows with the most excess bits to list.
    """
    if not hasattr(model, "target_logprobs"):
        raise TypeError(
            f"student_log_likelihood needs a model with target_logprobs, e.g. "
            f"LocalModel; got {type(model).__name__}"
        )
    report = LikelihoodReport(
        model_id=str(getattr(model, "model_id", getattr(model, "_model_id", ""))),
        target_field=target_field,
    )
    scores = _target_scores(
        rows, model=model, prompt_field=prompt_field, target_field=target_field
    )
    report.rows = [_row(i, s) for i, s in enumerate(scores)]
    picked = _evenly_spaced(len(rows), max(0, own_samples))
    if picked:
        prompts = [
            get_field(rows[i], prompt_field, "student_log_likelihood") for i in picked
        ]
        own, report.own_hit_max_tokens = _own_scores(
            prompts, model=model, max_tokens=max_tokens
        )
        report.own = [_row(i, s) for i, s in zip(picked, own)]
        rates = [r.bits_per_token for r in report.own if r.bits_per_token is not None]
        if rates:
            report.own_bits_per_token = statistics.median(rates)
            for r in report.rows:
                r.excess_bits = r.bits - r.tokens * report.own_bits_per_token
    console = _console()
    _emit(_renderable(report, rows, prompt_field, top_k, width=console.width), console)
    return report


def render_likelihood_report(
    report: LikelihoodReport,
    rows: list[dict] | None = None,
    prompt_field: str = "prompt",
    top_k: int = 5,
) -> str:
    """Render a ``student_log_likelihood`` report as plain text."""
    return _plain(_renderable(report, rows, prompt_field, top_k, width=WIDTH))


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _num(x: float | None, digits: int = 0) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}" if digits else fmt_num(round(x))


def _summary_cells(
    label: str, group: list[RowLikelihood], total: bool
) -> list[list[Text]]:
    """The table's lines for one group: all tokens, then reasoning and answer."""
    lines = [
        [
            Text(label),
            Text(f"{len(group):,}"),
            Text(_num(_median([r.tokens for r in group]))),
            Text(_num(_median([r.bits / r.tokens for r in group if r.tokens]), 2)),
            Text(_num(_median([r.bits for r in group]))),
            Text(fmt_num(sum(r.bits for r in group)) if total else "-"),
        ]
    ]
    thinking = [r for r in group if r.reasoning_tokens]
    if thinking:
        answer_rates = [
            (r.bits - r.reasoning_bits) / (r.tokens - r.reasoning_tokens)
            for r in thinking
            if r.tokens > r.reasoning_tokens
        ]
        for name, tokens, rates in (
            (
                "  reasoning",
                [r.reasoning_tokens for r in thinking],
                [r.reasoning_bits / r.reasoning_tokens for r in thinking],
            ),
            (
                "  answer",
                [r.tokens - r.reasoning_tokens for r in thinking],
                answer_rates,
            ),
        ):
            lines.append(
                [
                    Text(name, style="dim"),
                    Text(f"{len(thinking):,}", style="dim"),
                    Text(_num(_median(tokens)), style="dim"),
                    Text(_num(_median(rates), 2), style="dim"),
                    Text(""),
                    Text(""),
                ]
            )
    return lines


def _renderable(
    report: LikelihoodReport,
    rows: list[dict] | None,
    prompt_field: str,
    top_k: int,
    *,
    width: int,
) -> RenderableType:
    head = Text()
    head.append(f"{len(report.rows):,} rows", style="bold")
    head.append(f' · "{_clean(report.target_field)}" scored by {report.model_id}')
    body: list[RenderableType] = [
        head,
        Text("bits = -log2 of each token's probability under the student", style="dim"),
    ]
    if not report.rows:
        return Group(
            _rule("inspect", "student_log_likelihood"),
            Padding(Group(*body), (0, 0, 0, 2)),
        )
    table_rows = _summary_cells("targets", report.rows, total=True)
    if report.own:
        table_rows += _summary_cells("student's own", report.own, total=False)
    headers = ["", "rows", "median tokens", "bits/token", "bits/row", "total bits"]
    body += [Text(""), _fit_table(headers, table_rows, width - 2, left=set())]
    notes: list[str] = []
    truncated = sum(r.truncated for r in report.rows)
    if truncated:
        notes.append(
            f"{truncated:,} targets did not fit in the context; "
            "their ends are not scored"
        )
    if report.own_hit_max_tokens:
        notes.append(
            f"{report.own_hit_max_tokens:,} of the student's own responses "
            "hit max_tokens"
        )
    if report.own_bits_per_token is not None:
        excess = sorted(r.excess_bits for r in report.rows)
        q = lambda f: excess[min(len(excess) - 1, int(f * len(excess)))]  # noqa: E731
        line = Text()
        line.append(
            "excess over the student's own rate "
            f"({report.own_bits_per_token:.2f} bits/token): "
        )
        line.append(
            f"{fmt_num(round(report.total_excess_bits))} bits in total", style="bold"
        )
        spread = Text(
            f"  per row: p10 {fmt_num(round(q(0.1)))} · median {fmt_num(round(q(0.5)))}"
            f" · p90 {fmt_num(round(q(0.9)))} · max {fmt_num(round(excess[-1]))}"
        )
        body += [Text(""), line, spread]
        worst = sorted(report.rows, key=lambda r: -r.excess_bits)[: max(0, top_k)]
        if worst:
            body.append(Text("most excess bits:", style="dim"))
            for r in worst:
                item = Text(f"  row {r.row}  ")
                item.append(f"{fmt_num(round(r.excess_bits))}", style="bold")
                item.append(
                    f" ({fmt_num(r.tokens)} tokens × {r.bits_per_token or 0:.2f} bits)",
                    style="dim",
                )
                if rows is not None and r.row < len(rows):
                    prompt = (
                        rows[r.row].get(prompt_field)
                        if isinstance(rows[r.row], dict)
                        else None
                    )
                    if isinstance(prompt, list) and prompt:
                        prompt = prompt[-1].get("content", "")
                    if prompt:
                        item.append("  " + _clean(str(prompt)), style="dim")
                item.truncate(width - 2, overflow="ellipsis")
                body.append(item)
    body += [Text(n, style="dim") for n in notes]
    return Group(
        _rule("inspect", "student_log_likelihood"), Padding(Group(*body), (0, 0, 0, 2))
    )
