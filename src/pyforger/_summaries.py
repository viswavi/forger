"""Summaries of long sample values in ``@step`` reports.

A sample value too long for the report is summarized by its shape instead of
being cut blindly. These summaries are free and deterministic::

    code    def solve(n): ⏎ seen = set() … (312 lines, 9.8k chars, python)
    list    [5 items] a.py, b.py, c.py, …
    dict    {4 keys} name: Ada, files: [5 items], meta: {2 keys}, …
    prose   The patient presents with … and was discharged. (1,204 words, 7.9k chars)

JSON in a string is summarized as the list or dict it holds.

With a report model (``pyforger.steps.set_report_model(model)`` or
``FORGER_REPORT_MODEL=<litellm model id>``), the report also gets a one-sentence
LLM summary of each long value in the rows it shows, written with the step's
name, docstring, and field name as context. Those calls are cached on disk
(op ``report.summarize``), run outside every step and pipeline so they never
show up in a report's ops, tokens, or cost, and fall back silently to the
structural summary on any failure.
"""

from __future__ import annotations

import itertools
import json
import os
import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pyforger import _tracking
from pyforger._tracking import FieldSummary, StepRecord
from pyforger.op import op
from pyforger.utils import get_formatted_logger

logger = get_formatted_logger("forger.report")

LINE_BREAK = " ⏎ "  # shows where a single-line summary had a newline
_JSON_MAX_CHARS = 2_000_000  # longer strings are not parsed as JSON
_SCAN_LINES = 200  # lines read to tell code from prose
_FULL_LINES = 20  # lines of code or items shown in the full view
_FULL_LINE_CHARS = 160  # per line of the full view
_BRIEF_CHARS = 28  # per item or value in a list / dict preview

_MAX_SUMMARY_CALLS = 8  # LLM summaries per printed report
_SUMMARY_INPUT_CHARS = 6000  # value text an LLM summary reads
_SUMMARY_DOC_CHARS = 1000  # step docstring text an LLM summary reads
_SUMMARY_CHARS = 200  # longest LLM summary kept
_SUMMARY_WORKERS = 4

_FENCE = re.compile(r"^\s*```\s*([\w+#.-]*)\s*$")
_CODE_LINE = re.compile(
    r"^\s*(def |class |import |from [\w.]+ import |return\b|@\w|#include|#define"
    r"|using namespace|public |private |static |func |fn |let |const |var "
    r"|function\b|package |SELECT\b|INSERT\b|CREATE\b|if\s*\(|for\s*\(|while\s*\("
    r"|[}\])];?\s*$)"
    r"|[;{]\s*$|\)\s*:\s*$|^\s*(if|elif|else|for|while|try|except|with)\b.*:\s*$"
)
_SKIP_LINE = re.compile(
    r"^\s*(import |from [\w.]+ import |#include|using namespace|package |```"
    r"|[{}()\[\];,]*\s*$)"
)
_LANGUAGES = [
    (
        "python",
        re.compile(r"^\s*(def \w+\(|class \w+.*:\s*$|from [\w.]+ import )", re.M),
    ),
    ("c++", re.compile(r"#include\s*<(iostream|vector|string|bits/)|std::|cout\s*<<")),
    ("c", re.compile(r"#include\s*<\w+\.h>")),
    ("java", re.compile(r"public\s+(static\s+)?(class|void)|System\.out\.")),
    ("go", re.compile(r"^package \w+|^func \w+\(", re.M)),
    ("rust", re.compile(r"^\s*(pub )?fn \w+\(|let mut ", re.M)),
    ("javascript", re.compile(r"\bfunction\b|=>|console\.log|^\s*(const|let) ", re.M)),
    ("sql", re.compile(r"\bSELECT\b[\s\S]*\bFROM\b", re.I)),
    ("python", re.compile(r"^\s*(import \w+|print\(|return\b)", re.M)),
]


# --------------------------------------------------------------------------- #
# Structural summaries
# --------------------------------------------------------------------------- #


def fmt_size(n: int) -> str:
    """Compact count: ``640``, ``9.8k``, ``1.2M``."""
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}k"
    return str(n)


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def _one_line(text: str) -> str:
    """Collapse runs of spaces and mark line breaks: ``a ⏎ b``."""
    lines = (" ".join(line.split()) for line in text.splitlines())
    return LINE_BREAK.join(line for line in lines if line)


def _cut(text: str, n: int) -> str:
    """``text`` cut to ``n`` characters with … (at a word break when one is near)."""
    if len(text) <= n:
        return text
    if n <= 1:
        return "…"[:n]
    head = text[: n - 1]
    space = head.rfind(" ")
    if space >= n * 0.6:
        head = head[:space]
    return head.rstrip() + "…"


def _cut_tail(text: str, n: int) -> str:
    """The last ``n`` characters of ``text``, starting with … (at a word break)."""
    if len(text) <= n:
        return text
    tail = text[len(text) - n + 1 :]
    space = tail.find(" ")
    if 0 <= space <= n * 0.4:
        tail = tail[space + 1 :]
    return "…" + tail.lstrip()


def _head_tail(text: str, n: int, sep: str = " … ") -> str:
    """The start and end of ``text`` in about ``n`` characters (``The … end.``)."""
    if len(text) <= n:
        return text
    head = _cut(text, n * 2 // 3).removesuffix("…").rstrip()
    tail = _cut_tail(text, max(1, n - len(head) - len(sep))).removeprefix("…")
    return head + sep + tail.lstrip()


def parse_json(value: Any) -> Any:
    """The list or dict a JSON string holds, or None."""
    if not isinstance(value, str) or len(value) > _JSON_MAX_CHARS:
        return None
    text = value.strip()
    if not (text[:1], text[-1:]) in (("{", "}"), ("[", "]")):
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    return parsed if isinstance(parsed, (dict, list)) else None


def code_language(text: str) -> str | None:
    """``python``, ``c++``, ... when ``text`` looks like code, ``code`` if unsure.

    None for prose. Reads the first 200 lines: code has a fence or at least a
    third of its non-blank lines look like statements (``def f():``, ``x = 1;``).
    """
    lines = text.splitlines()[:_SCAN_LINES]
    fence = next((m for m in map(_FENCE.match, lines) if m), None)
    nonblank = [line for line in lines if line.strip()]
    if fence is None:
        if len(nonblank) < 2:
            return None
        codeish = sum(1 for line in nonblank if _CODE_LINE.search(line))
        indented = sum(1 for line in nonblank if line.startswith(("    ", "\t")))
        if codeish < 2 or codeish + indented / 2 < len(nonblank) / 3:
            return None
    if fence is not None and fence.group(1):
        return fence.group(1).lower()
    if lines and lines[0].startswith("#!"):
        words = lines[0][2:].rsplit("/", 1)[-1].split()  # "#!/usr/bin/env python3"
        return words[-1] if words else "shell"
    head = "\n".join(lines)
    for name, pattern in _LANGUAGES:
        if pattern.search(head):
            return name
    return "code"


def _brief(value: Any, n: int = _BRIEF_CHARS) -> str:
    """A short inline preview: the value itself when it fits, else its size."""
    if isinstance(value, str):
        return _cut(_one_line(value), n)
    if isinstance(value, (dict, list, tuple)) and len(value) > n:
        text = ""  # too many entries to fit: skip printing them
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = repr(value)
    if text and len(text) <= n:
        return text
    if isinstance(value, dict):
        return "{" + _plural(len(value), "key") + "}"
    if isinstance(value, (list, tuple)):
        return "[" + _plural(len(value), "item") + "]"
    return _cut(text, n)


def _fill(head: str, parts: Iterable[str], count: int, limit: int) -> str:
    """``head`` and as many of the ``count`` parts as fit in ``limit``, then ``…``."""
    text = head
    for i, part in enumerate(parts):
        joiner = " " if i == 0 else ", "
        more = ", …" if i < count - 1 else ""
        if len(text) + len(joiner) + len(part) + len(more) > limit:
            return text + joiner + "…"
        text += joiner + part
    return text


def _list_shape(value: list | tuple) -> str:
    return "[" + _plural(len(value), "item") + "]"


def _dict_shape(value: dict) -> str:
    return "{" + _plural(len(value), "key") + "}"


def _text_stats(text: str, language: str | None) -> str:
    """``312 lines, 9.8k chars, python`` or ``1,204 words, 7.9k chars``."""
    n_lines = text.count("\n") + 1
    parts = []
    if language is not None or n_lines > 1:
        parts.append(_plural(n_lines, "line"))
    if language is None:
        parts.append(_plural(len(text.split()), "word"))
    parts.append(f"{fmt_size(len(text))} chars")
    if language is not None:
        parts.append(language)
    return ", ".join(parts)


def shape_of(value: Any) -> str:
    """One line on the size and kind of ``value``, e.g. ``[5 items]``."""
    parsed = parse_json(value)
    if parsed is not None:
        return "JSON " + shape_of(parsed)
    if isinstance(value, dict):
        return _dict_shape(value)
    if isinstance(value, (list, tuple)):
        return _list_shape(value)
    if isinstance(value, str):
        return _text_stats(value, code_language(value))
    return type(value).__name__


def _meaningful_lines(text: str) -> list[str]:
    """Code lines worth previewing: no blanks, imports, fences, or lone brackets."""
    return [line for line in text.splitlines() if not _SKIP_LINE.match(line)]


def compact_summary(value: Any, limit: int) -> str | None:
    """A one-line summary of ``value`` in about ``limit`` characters.

    None when the value has no shape to summarize (e.g. a long number).
    """
    parsed = parse_json(value)
    if parsed is not None:
        inner = compact_summary(parsed, limit - 5)
        return None if inner is None else "JSON " + inner
    if isinstance(value, dict):
        parts = (f"{_one_line(str(k))}: {_brief(v)}" for k, v in value.items())
        return _fill(_dict_shape(value), parts, len(value), limit)
    if isinstance(value, (list, tuple)):
        items = (_brief(v) for v in value)
        return _fill(_list_shape(value), items, len(value), limit)
    if not isinstance(value, str):
        return None
    language = code_language(value)
    tag = f" ({_text_stats(value, language)})"
    room = max(10, limit - len(tag) - 2)
    if language is not None:
        lines = [" ".join(line.split()) for line in _meaningful_lines(value)]
        return _cut(LINE_BREAK.join(lines), room) + tag
    return _head_tail(_one_line(value), room) + tag


def full_summary(value: Any, limit: int) -> str | None:
    """A multi-line summary of ``value`` in about ``limit`` characters."""
    parsed = parse_json(value)
    if parsed is not None:
        inner = full_summary(parsed, limit)
        return None if inner is None else "JSON " + inner
    if isinstance(value, (dict, list, tuple)):
        if isinstance(value, dict):
            head = _dict_shape(value)
            items = (f"{_one_line(str(k))}: {_brief(v, 100)}" for k, v in value.items())
        else:
            head = _list_shape(value)
            items = (f"- {_brief(v, 100)}" for v in value)
        lines, size = [head], len(head)
        for item in itertools.islice(items, _FULL_LINES):
            item = _cut(item, _FULL_LINE_CHARS)
            if size + len(item) > limit:
                break
            lines.append(item)
            size += len(item) + 1
        if len(lines) - 1 < len(value):
            lines.append(f"… {len(value) - len(lines) + 1:,} more")
        return "\n".join(lines)
    if not isinstance(value, str):
        return None
    language = code_language(value)
    tag = f"({_text_stats(value, language)})"
    if language is not None:
        lines, size = [], 0
        for line in value.strip("\n").splitlines()[:_FULL_LINES]:
            line = _cut(line.rstrip(), _FULL_LINE_CHARS)
            if size + len(line) > limit:
                break
            lines.append(line)
            size += len(line) + 1
        return "\n".join(lines + [f"… {tag}"])
    return _head_tail(value.strip(), limit, sep=" …\n… ") + "\n" + tag


def summary_source(value: Any) -> str:
    """The text an LLM summary reads: the value, clipped to 6,000 characters."""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, indent=1, default=str)
        except (TypeError, ValueError):
            text = repr(value)
    if len(text) <= _SUMMARY_INPUT_CHARS:
        return text
    head = text[: _SUMMARY_INPUT_CHARS * 2 // 3]
    tail = text[-(_SUMMARY_INPUT_CHARS // 3) :]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n… [{omitted:,} characters omitted] …\n{tail}"


# --------------------------------------------------------------------------- #
# LLM summaries
# --------------------------------------------------------------------------- #


_UNSET: Any = object()
_REPORT_MODEL: list[Any] = [_UNSET]  # set_report_model's model, or _UNSET
_ENV_MODELS: dict[str, Any] = {}  # FORGER_REPORT_MODEL id -> Model (None: failed)


def set_report_model(model: Any) -> None:
    """Write a one-sentence LLM summary of each long sample value in step reports.

    ``model`` is any ``Model``; ``None`` turns LLM summaries off, even when
    ``FORGER_REPORT_MODEL`` is set. Each report makes at most 8 calls, which
    are cached on disk, so printing a report again is free.
    """
    _REPORT_MODEL[0] = model


def report_model() -> Any:
    """The model for LLM summaries: ``set_report_model``'s, else the env's, else None.

    ``FORGER_REPORT_MODEL=<litellm model id>`` builds a ``Model`` with the key in
    ``LITELLM_API_KEY``.
    """
    if _REPORT_MODEL[0] is not _UNSET:
        return _REPORT_MODEL[0]
    model_id = os.environ.get("FORGER_REPORT_MODEL", "").strip()
    if not model_id:
        return None
    if model_id not in _ENV_MODELS:
        from pyforger.acquire import model as model_module

        try:
            _ENV_MODELS[model_id] = model_module.Model(
                model_id,
                api_key=os.environ.get("LITELLM_API_KEY"),
                num_retries=1,
                timeout=60,
            )
        except Exception as exc:  # logged once: the failure is cached
            logger.warning(
                "FORGER_REPORT_MODEL=%s is unusable, so step reports get no LLM "
                "summaries: %s",
                model_id,
                exc,
            )
            _ENV_MODELS[model_id] = None
    return _ENV_MODELS[model_id]


_HINTED = [False]  # the "no report model" hint is logged once per process


def _hint_report_model() -> None:
    """Once per process: say how to get LLM summaries of long values.

    Not when ``set_report_model(None)`` turned them off on purpose, nor when
    ``FORGER_REPORT_MODEL`` names a model that failed (``report_model`` said so).
    """
    if (
        _HINTED[0]
        or _REPORT_MODEL[0] is None
        or os.environ.get("FORGER_REPORT_MODEL", "").strip()
    ):
        return
    _HINTED[0] = True
    logger.warning(
        "step reports summarize long values by their shape only (e.g. "
        "'def solve(n): … (312 lines, python)'). For a one-sentence summary of "
        "what each says, call pyforger.steps.set_report_model(Model(...)) or set "
        "FORGER_REPORT_MODEL=<litellm model id>, e.g. a cheap model with no "
        "reasoning: at most 8 cached calls per report. set_report_model(None) "
        "silences this."
    )


def summary_prompt(step: str, doc: str, field: str, shape: str, text: str) -> str:
    """The prompt asking for one sentence on ``text``, the ``field`` of a row."""
    doc = doc.strip()
    if len(doc) > _SUMMARY_DOC_CHARS:
        doc = doc[:_SUMMARY_DOC_CHARS] + " …"
    return (
        "You are helping someone review the output of a data pipeline. Below is "
        f'one field of one output row of the step "{step}".\n\n'
        f"What the step does:\n{doc or '(no description)'}\n\n"
        f"Field: {field}\nShape: {shape}\n\n<value>\n{text}\n</value>\n\n"
        "In one short sentence (under 25 words), say what this value contains so "
        "the reviewer can judge it without reading it. Describe its content, not "
        "its formatting. Reply with the sentence only."
    )


@op(name="report.summarize")
def _summarize(model: Any, prompt: str) -> str:
    return str(model.generate(prompt)[0])


def _first_sentence(text: str) -> str:
    from pyforger.acquire.model import split_reasoning

    # A reasoning model may put "<think> … </think>" before its sentence.
    text = split_reasoning(text)[1]
    line = next((x.strip() for x in text.strip().splitlines() if x.strip()), "")
    line = line.strip("\"'` ")
    return _cut(line, _SUMMARY_CHARS)


def _summarize_one(model: Any, prompt: str) -> str:
    """One LLM summary, or "" on any failure (logged at debug level)."""
    try:
        return _first_sentence(_summarize(model, prompt))
    except Exception as exc:  # reports never fail because of a summary
        logger.debug("report summary failed: %s: %s", type(exc).__name__, exc)
        return ""


def add_llm_summaries(rec: StepRecord, rows: int | None) -> None:
    """Fill in ``llm`` for long values in the first ``rows`` sample rows.

    ``rows`` counts per output stream; None means every stored sample row.
    Without a report model it does nothing, but the first time a report has a
    value it could have summarized, it logs how to turn summaries on (unless
    ``set_report_model(None)`` turned them off). Values summarized before (in
    this record) are skipped; a failed summary is left empty and not retried.
    """
    try:
        jobs: list[tuple[FieldSummary, str]] = []
        samples = [(rec.sample, rec.sample_summaries)] + [
            (s.sample, s.sample_summaries) for s in rec.streams or []
        ]
        pairs = [
            pair
            for sample, summaries in samples
            for pair in zip(sample[: None if rows is None else max(0, rows)], summaries)
        ]
        for row, fields in pairs:
            keys = list(row) if isinstance(row, dict) else []
            for j, summary in sorted(fields.items()):
                if summary.llm is not None or not summary.source or j >= len(keys):
                    continue
                prompt = summary_prompt(
                    rec.name, rec.doc, str(keys[j]), summary.shape, summary.source
                )
                jobs.append((summary, prompt))
        jobs = jobs[:_MAX_SUMMARY_CALLS]
        if not jobs:
            return
        model = report_model()
        if model is None:
            _hint_report_model()
            return

        def run(prompt: str) -> str:
            # Outside every step, pipeline, and op, with usage not counted.
            return _tracking.run_uncounted(_summarize_one, model, prompt)

        with ThreadPoolExecutor(max_workers=min(_SUMMARY_WORKERS, len(jobs))) as pool:
            results = list(pool.map(run, [prompt for _, prompt in jobs]))
        for (summary, _), text in zip(jobs, results):
            summary.llm = text
    except Exception as exc:
        logger.debug("report summaries skipped: %s: %s", type(exc).__name__, exc)
