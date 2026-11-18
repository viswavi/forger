"""Repeated phrases and names per field: templating and name collapse."""

from __future__ import annotations

from pydantic import BaseModel, Field
from rich.console import Group, RenderableType
from rich.padding import Padding
from rich.text import Text

from pyforger import _phrases
from pyforger._phrases import FieldPhrases, Phrase
from pyforger._render import WIDTH, _clean, _console, _emit, _plain, _rule
from pyforger.op import tracked


class PhraseReport(BaseModel):
    """What ``phrase_frequency`` found: per string field, its repeated phrases.

    ``n_rows`` is the number of rows given and ``scanned`` how many of them
    (evenly spaced) were read.
    """

    n_rows: int
    scanned: int
    fields: list[FieldPhrases] = Field(default_factory=list)


@tracked("inspect.phrase_frequency")
def phrase_frequency(
    rows: list[dict],
    fields: list[str] | str | None = None,
    *,
    ngram_range: tuple[int, int] = (2, 4),
    top_k: int = 10,
    min_share: float = 0.05,
    max_rows: int = 2000,
    max_chars: int = 2000,
) -> PhraseReport:
    """Print and return the phrases and names that the most rows share, per field.

    Counts document frequency: the share of rows whose value contains a word
    n-gram (lowercased) or a capitalized name, e.g. ``rewrite "i need" 78/79``
    or ``context Maya 46/79``. Rows can all be distinct, and pass exact-match
    and ROUGE-L filters, while sharing a template or a name. No LLM is used.
    Markup, headings and the sections under headings most rows share, lines
    most rows share, code, categorical fields, and stopword-only phrases are
    left out (see ``pyforger._phrases``). Rows are not modified.

    Args:
        rows: The rows (dicts).
        fields: Fields to count; default: every field with string values or
            chat messages (counted by their contents).
        ngram_range: Smallest and largest phrase length in words; ``(1, 4)``
            also counts single words.
        top_k: Phrases and names listed per field.
        min_share: Leave out phrases in fewer than this share of rows.
        max_rows: Read at most this many rows, evenly spaced.
        max_chars: Read at most this many characters of each value.
    """
    lo, hi = ngram_range
    if not 1 <= lo <= hi:
        raise ValueError(
            f"ngram_range must be (lo, hi) with 1 <= lo <= hi, got {ngram_range}"
        )
    scanned = _phrases.evenly_spaced(list(rows), max_rows)
    if fields is None:
        names = [
            k
            for k in dict.fromkeys(k for r in scanned if isinstance(r, dict) for k in r)
            if any(_phrases.is_text(r.get(k)) for r in scanned if isinstance(r, dict))
        ]
    else:
        names = [fields] if isinstance(fields, str) else list(fields)
    report = PhraseReport(n_rows=len(rows), scanned=len(scanned))
    for name in names:
        report.fields.append(
            _phrases.field_phrases(
                _phrases.field_texts(scanned, name),
                str(name),
                ngram_range=(lo, hi),
                top_k=top_k,
                min_share=min_share,
                max_chars=max_chars,
            )
        )
    console = _console()
    _emit(_report_renderable(report, width=console.width), console)
    return report


def render_phrase_report(report: PhraseReport) -> str:
    """Render a ``phrase_frequency`` report as plain text."""
    return _plain(_report_renderable(report, width=WIDTH))


def _items(items: list[Phrase], n: int) -> str:
    return " · ".join(f'"{_clean(p.text)}" {p.rows}/{n} ({p.share:.0%})' for p in items)


def _report_renderable(report: PhraseReport, *, width: int) -> RenderableType:
    rows = f"{report.n_rows:,} rows"
    if report.scanned < report.n_rows:
        rows += f" ({report.scanned:,} read, evenly spaced)"
    body: list[RenderableType] = [Text(rows, style="bold")]
    for f in report.fields:
        head = Text()
        head.append(_clean(f.field), style="bold")
        if f.skipped is not None:
            head.append(f"  skipped: {f.skipped}", style="dim")
            body.append(head)
            continue
        head.append(f"  {f.n_rows:,} values · median {f.words:,} words", style="dim")
        body.append(Text(""))
        body.append(head)
        if not f.phrases and not f.names:
            body.append(Padding(Text("nothing repeated", style="dim"), (0, 0, 0, 2)))
        if f.phrases:
            line = Text()
            line.append("phrases: ", style="dim")
            line.append(_items(f.phrases, f.n_rows))
            body.append(Padding(line, (0, 0, 0, 2)))
        if f.names:
            line = Text()
            line.append("names: ", style="dim")
            line.append(_items(f.names, f.n_rows))
            body.append(Padding(line, (0, 0, 0, 2)))
    return Group(
        _rule("inspect", "phrase_frequency"), Padding(Group(*body), (0, 0, 0, 2))
    )
