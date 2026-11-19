"""Repeated phrases and names across rows: templating and name collapse.

LLM-written rows often share a phrase (``"I need"`` in 78 of 79 rewrites) or
a name (the writer is ``Maya`` in 46 of 79 contexts) while every row is
distinct, so exact-match and ROUGE-L filters pass them all. This counts, per
string field, the share of rows (document frequency) that contain each word
n-gram and each capitalized name, cheaply and without an LLM. It backs
``inspect.phrase_frequency`` and the ``repeated:`` line of step reports.

Text a template put there is not collapse, so it is left out before counting:

- markup: fenced code blocks, inline code, HTML tags, and URLs;
- headings (``## Input``, ``Input format:``, ``**Output**``, a short line such
  as ``Constraints``) and, when most rows have the same heading, the section
  under it: an ``Input`` section reads "The first line contains" in every
  statement because the format asks for it;
- boilerplate lines: a line that is the same in at least ``BOILERPLATE_SHARE``
  of the rows (an instruction prefix, a fixed footer);
- code: values where at least 40% of the lines look like code (``import
  sys``, ``def solve():``); a field of mostly code is skipped;
- categorical fields, where fewer than half the values are distinct (a
  ``topic`` picked from a list): their repeats are by design; and identifier
  fields, one word per value (``imp_s1_003``).

Phrases of stopwords only (``"if you"``, ``"of the"``), a determiner and a
noun (``"the problem"``: the subject, not a way of phrasing it), and phrases
across a number are skipped, and a phrase inside a longer one that is in
nearly as many rows (``"need you"`` inside ``"i need you to"``) is left out.
In the step report, the input rows' text is a template too: a phrase they
had nearly as often is not the step's doing. Names are
capitalized words seen mid-sentence and rarely in lower case, possessives
folded (``Maya's`` counts as ``Maya``), weekdays and months left out.
Nothing here imports the rest of pyforger.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field

from pyforger.utils import as_text, is_chat

# A line, or a heading, in at least this share of rows is template text.
BOILERPLATE_SHARE = 0.8
# The step report's repeated: line. A phrase in more than PHRASE_SHARE of a
# field's rows (more for long values, see repeat_bar), or a name in more than
# NAME_SHARE, is printed; at most MAX_REPEATS items, from up to
# STEP_SCAN_ROWS rows of STEP_SCAN_CHARS characters each.
PHRASE_SHARE = 0.3
NAME_SHARE = 0.3
MAX_REPEATS = 3
MIN_STEP_ROWS = 10
STEP_SCAN_ROWS = 200
STEP_SCAN_CHARS = 1000
# A phrase is left out when a longer one containing it is in at least this
# share of its rows: "i need" (78 rows) stays; "need" (78 rows) goes.
SUBSUMED_SHARE = 0.9

STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can can't cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he he'd he'll he's
    her here here's hers herself him himself his how how's i i'd i'll i'm i've
    if in into is isn't it it's its itself just let's me more most mustn't my
    myself no nor not now of off on once only or other ought our ours ourselves
    out over own same shan't she she'd she'll she's should shouldn't so some
    such than that that's the their theirs them themselves then there there's
    these they they'd they'll they're they've this those through to too under
    until up very was wasn't we we'd we'll we're we've were weren't what what's
    when when's where where's which while who who's whom why why's will with
    won't would wouldn't you you'd you'll you're you've your yours yourself
    yourselves also get got really even still ok okay yes yeah
    """.split()
)
_DETERMINERS = frozenset(
    "a an the this that these those each every any some my your our their his "
    "her its".split()
)
_NOT_NAMES = frozenset(
    """
    monday tuesday wednesday thursday friday saturday sunday january february
    march april may june july august september october november december
    """.split()
)

_FENCE = re.compile(r"^[ \t]*```.*?(?:^[ \t]*```[^\n]*$|\Z)", re.M | re.S)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_TAG = re.compile(r"</?[A-Za-z][^>\n]*>")
_URL = re.compile(r"https?://\S+")
# "## Input", "Input format:", "**Output**", "Example 1:", "Constraints"
_HEADING = re.compile(
    r"^(?:#{1,6}\s+(?P<md>.+)|\*\*(?P<bold>[^*]{1,40})\*\*:?"
    r"|(?P<label>[\w ()/,.'-]{1,40}):|(?P<short>[A-Z][\w-]*(?: [\w-]+){0,2}))$"
)
_CODE_LINE = re.compile(
    r"^\s*(?:def |class |import |from \S+ import |return\b|elif\b|else:|try:|"
    r"except\b|for .+ in .+:|while .+:|if .+:|print\(|#include|int main|public |"
    r"[{}])|[;{}]\s*$|^\s*\w+(?:\[.*\])? ?[-+*/]?= "
)
_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_SENTENCE_END = re.compile(r"[.!?:;\"“]")


class Phrase(BaseModel):
    """A phrase (lowercased) or a name, and how many rows contain it."""

    text: str
    kind: Literal["phrase", "name"] = "phrase"
    rows: int
    share: float


class FieldPhrases(BaseModel):
    """The most repeated phrases and names of one string field.

    ``n_rows`` is how many of the scanned rows have a string value there.
    ``skipped`` says why a field was not counted, e.g. ``"code"`` or
    ``"categorical (12 distinct of 191)"``.
    """

    field: str
    n_rows: int
    words: int = 0  # median words per value (the whole value)
    phrases: list[Phrase] = Field(default_factory=list)
    names: list[Phrase] = Field(default_factory=list)
    skipped: str | None = None


def evenly_spaced(items: list[Any], limit: int) -> list[Any]:
    """At most ``limit`` items, evenly spaced, in order."""
    if len(items) <= limit:
        return list(items)
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def looks_like_code(text: str) -> bool:
    """True when 40% of the first 40 non-empty lines (2 or more) look like code."""
    lines = [line for line in text[:4000].splitlines() if line.strip()][:40]
    if len(lines) < 2:
        return False
    return sum(1 for line in lines if _CODE_LINE.search(line)) >= 0.4 * len(lines)


def _heading(line: str) -> str | None:
    """The normalized heading of a heading line, e.g. ``input``; else None."""
    m = _HEADING.match(line)
    if m is None:
        return None
    text = next(g for g in m.groups() if g is not None)
    # "Example 1" and "Example 2" are one heading.
    return re.sub(r"[\d\s:#*]+$", "", text.lower()).strip() or text.lower()


def _sections(text: str) -> list[tuple[str | None, list[str]]]:
    """``(heading, prose lines)`` of ``text``, markup removed."""
    text = _URL.sub(" ", _TAG.sub(" ", _INLINE_CODE.sub(" ", _FENCE.sub("\n", text))))
    out: list[tuple[str | None, list[str]]] = [(None, [])]
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        heading = _heading(line)
        if heading is not None:
            out.append((heading, []))
        else:
            out[-1][1].append(line)
    return out


def _norm(line: str) -> str:
    return " ".join(line.lower().split())


def _tokens(line: str) -> list[tuple[str, bool]]:
    """``(word, sentence-initial)`` pairs; numbers become ``""`` (a break)."""
    out: list[tuple[str, bool]] = []
    last = 0
    for m in _WORD.finditer(line):
        # The line's first word, or the first after ". " / "? " / ': "'.
        initial = not out or bool(_SENTENCE_END.search(line[last : m.start()]))
        word = m.group()
        out.append(("" if word.isdigit() else word, initial))
        last = m.end()
    return out


def _ngrams(words: list[str], lo: int, hi: int) -> set[str]:
    """The n-grams of ``words`` for n in [lo, hi] that can be phrases.

    Not across a number (``""``), with a word of three or more letters that
    is not a stopword, and not a determiner and a noun: ``"the problem"``
    names the subject, it is not a way of phrasing.
    """
    found: set[str] = set()
    content = [len(w) >= 3 and w not in STOPWORDS for w in words]
    for i, first in enumerate(words):
        has_content = False
        for j in range(i, min(i + hi, len(words))):
            if not words[j]:
                break
            has_content = has_content or content[j]
            n = j - i + 1
            if n >= lo and has_content and not (n == 2 and first in _DETERMINERS):
                found.add(" ".join(words[i : j + 1]))
    return found


def _name(word: str) -> str | None:
    """``Maya`` for ``Maya`` or ``Maya's``; None for words that are not names."""
    if word.endswith("'s"):
        word = word[:-2]
    if len(word) < 2 or not (word[0].isupper() and word[1:].islower()):
        return None
    low = word.lower()
    return None if low in STOPWORDS or low in _NOT_NAMES else word


class _Counts:
    """How many rows contain each phrase and each name."""

    def __init__(
        self, texts: list[str], ngram_range: tuple[int, int], max_chars: int
    ) -> None:
        self.n_rows = n = len(texts)
        rows = [_sections(t[:max_chars]) for t in texts]
        # Headings and lines in most rows are the template's, not collapse.
        heading_rows = Counter(h for secs in rows for h in {h for h, _ in secs if h})
        line_rows = Counter(
            line for secs in rows for line in {_norm(x) for _, ls in secs for x in ls}
        )
        least = BOILERPLATE_SHARE * n if n >= 5 else n + 1
        shared = {h for h, k in heading_rows.items() if k >= least}
        boiler = {line for line, k in line_rows.items() if k >= least}

        self.phrases: Counter[str] = Counter()
        capital: Counter[str] = Counter()  # rows with the name capitalized
        mid: Counter[str] = Counter()  # ... not at a sentence start
        lower: Counter[str] = Counter()  # rows with the word in lower case
        lo, hi = ngram_range
        for secs in rows:
            grams: set[str] = set()
            caps: set[str] = set()
            mids: set[str] = set()
            lows: set[str] = set()
            for heading, lines in secs:
                if heading in shared:
                    continue
                for line in lines:
                    if _norm(line) in boiler:
                        continue
                    toks = _tokens(line)
                    grams |= _ngrams([w.lower() for w, _ in toks], lo, hi)
                    for word, initial in toks:
                        name = _name(word)
                        if name is not None:
                            caps.add(name)
                            if not initial:
                                mids.add(name)
                        elif word.islower():
                            lows.add(word)
            self.phrases.update(grams)
            capital.update(caps)
            mid.update(mids)
            lower.update(lows)
        self.names = Counter(
            {w: k for w, k in capital.items() if mid[w] and lower[w.lower()] * 5 < k}
        )

    def top_phrases(self, min_rows: int, limit: int) -> list[tuple[str, int]]:
        """Phrases in at least ``min_rows`` rows, most rows first."""
        frequent = {p: k for p, k in self.phrases.items() if k >= min_rows}
        # The most rows of a longer frequent phrase containing each phrase.
        within: dict[str, int] = {}
        for p, k in frequent.items():
            words = p.split()
            for size in range(1, len(words)):
                for i in range(len(words) - size + 1):
                    sub = " ".join(words[i : i + size])
                    within[sub] = max(within.get(sub, 0), k)
        kept = [
            (p, k) for p, k in frequent.items() if within.get(p, 0) < SUBSUMED_SHARE * k
        ]
        kept.sort(key=lambda pk: (-pk[1], -len(pk[0]), pk[0]))
        return kept[:limit]

    def top_names(self, min_rows: int, limit: int) -> list[tuple[str, int]]:
        """Names in at least ``min_rows`` rows, most rows first."""
        names = [(w, k) for w, k in self.names.items() if k >= min_rows]
        names.sort(key=lambda wk: (-wk[1], wk[0]))
        return names[:limit]


def is_text(value: Any) -> bool:
    """True for a string or chat messages, e.g. ``[{"role": "user", ...}]``."""
    return isinstance(value, str) or is_chat(value)


def field_texts(rows: Iterable[Any], field: str) -> list[str]:
    """The text values of ``field`` in the dict rows; messages give their contents."""
    return [
        as_text(v)
        for r in rows
        if isinstance(r, dict)
        for v in [r.get(field)]
        if is_text(v)
    ]


def skip_reason(texts: list[str], code: list[bool] | None = None) -> str | None:
    """Why a field's values are not counted, or None to count them.

    ``code`` is ``looks_like_code`` of each text, when known.
    """
    if not texts:
        return "no string values"
    half = 0.5 * len(texts)
    distinct = len(set(texts))
    if distinct < half:
        return f"categorical ({distinct} distinct of {len(texts)})"
    if sum(1 for t in texts if len(t[:200].split()) <= 1) > half:
        return "identifiers"  # "imp_s1_003": one word each
    code = code if code is not None else [looks_like_code(t) for t in texts]
    if sum(code) > half:
        return "code"
    return None


def field_phrases(
    texts: list[str],
    field: str,
    *,
    ngram_range: tuple[int, int] = (2, 4),
    top_k: int = 10,
    min_share: float = 0.05,
    max_chars: int = 2000,
) -> FieldPhrases:
    """The top phrases and names of one field's values (see the module doc)."""
    code = [looks_like_code(t) for t in texts]
    out = FieldPhrases(field=field, n_rows=len(texts), skipped=skip_reason(texts, code))
    if out.skipped is not None:
        return out
    prose = [t for t, is_code in zip(texts, code) if not is_code]
    counts = _Counts(prose, ngram_range, max_chars)
    lengths = sorted(len(t.split()) for t in prose)
    out.words = lengths[len(lengths) // 2] if lengths else 0
    n = len(texts)
    min_rows = max(2, math.ceil(min_share * n - 1e-9))
    out.phrases = [
        Phrase(text=p, rows=k, share=k / n)
        for p, k in counts.top_phrases(min_rows, top_k)
    ]
    out.names = [
        Phrase(text=w, kind="name", rows=k, share=k / n)
        for w, k in counts.top_names(min_rows, top_k)
    ]
    return out


def repeat_bar(words: int) -> float:
    """The share of rows a phrase must be in to count as repeated.

    30% for values of up to 100 words; longer values contain more phrases by
    chance, so the bar rises with the median length (60% at 200 words), at
    most 90%. Names always need 30%.
    """
    return min(0.9, max(PHRASE_SHARE, PHRASE_SHARE * words / 100))


class Repeat(BaseModel):
    """One ``repeated:`` item of a step report: a phrase or name in a field."""

    field: str
    text: str
    kind: Literal["phrase", "name"] = "phrase"
    rows: int
    n_rows: int


def _haystack(texts: list[str]) -> list[tuple[str, str]]:
    """``(" words joined ", raw text)`` per text, for ``_share_in``."""
    out = []
    for t in texts:
        t = t[:STEP_SCAN_CHARS]
        words = " ".join(w.lower() for w, _ in _tokens(t))
        out.append((f" {words} ", t))
    return out


def _share_in(hay: list[tuple[str, str]], item: Phrase) -> float:
    """The share of ``hay`` rows that contain the phrase or name."""
    if not hay:
        return 0.0
    if item.kind == "name":
        pattern = re.compile(rf"\b{re.escape(item.text)}\b")
        hits = sum(1 for _, raw in hay if pattern.search(raw))
    else:
        needle = f" {item.text} "
        hits = sum(1 for words, _ in hay if needle in words)
    return hits / len(hay)


def step_repeats(rows: list[Any], input_rows: list[Any] | None) -> list[Repeat]:
    """The ``repeated:`` items of a step's output rows, at most ``MAX_REPEATS``.

    Looks at the string fields the step added or changed (all of them when
    the input rows are not dicts), in up to ``STEP_SCAN_ROWS`` evenly spaced
    rows of at least ``MIN_STEP_ROWS``. A phrase counts when it is in more
    than ``repeat_bar`` of a field's rows, a name in more than 30%, unless
    the input rows had it nearly as often (the template, or the seed, put it
    there: not this step). Each field gives its most repeated phrase, and the
    most repeated name gets a place.
    """
    dicts = [r for r in rows if isinstance(r, dict)]
    scanned = evenly_spaced(dicts, STEP_SCAN_ROWS)
    if len(scanned) < MIN_STEP_ROWS:
        return []
    before = [r for r in input_rows or [] if isinstance(r, dict)]
    before_scan = evenly_spaced(before, STEP_SCAN_ROWS)
    before_keys = {k for r in before for k in r}
    keys = list(dict.fromkeys(k for r in scanned for k in r))
    phrases: list[Repeat] = []
    names: list[Repeat] = []
    for key in keys:
        texts = field_texts(scanned, key)
        if len(texts) < MIN_STEP_ROWS:
            continue
        if key in before_keys:
            old = {v for v in field_texts(before, key)}
            if all(t in old for t in texts):
                continue  # passed through unchanged, e.g. by a filter
            base = field_texts(before_scan, key)
        else:  # a new field may restate any input field
            base = [
                "\n".join(as_text(v) for v in r.values() if is_text(v))
                for r in before_scan
            ]
        found = field_phrases(
            texts, str(key), top_k=5, min_share=PHRASE_SHARE, max_chars=STEP_SCAN_CHARS
        )
        hay = _haystack(base) if base else []
        n = len(texts)
        for items, bar, out in (
            (found.phrases, repeat_bar(found.words), phrases),
            (found.names, NAME_SHARE, names),
        ):
            for item in items:
                if item.share < bar or _share_in(hay, item) >= 0.8 * item.share:
                    continue
                out.append(
                    Repeat(
                        field=str(key),
                        text=item.text,
                        kind=item.kind,
                        rows=item.rows,
                        n_rows=n,
                    )
                )
                break  # the field's most repeated one
    phrases.sort(key=lambda r: -r.rows / r.n_rows)
    names.sort(key=lambda r: -r.rows / r.n_rows)
    best_name = names[:1]
    picked = phrases[: MAX_REPEATS - len(best_name)] + best_name
    if len(picked) < MAX_REPEATS:  # room for more names, one per name
        seen = {r.text for r in best_name}
        for r in names[1:]:
            if r.text not in seen and len(picked) < MAX_REPEATS:
                seen.add(r.text)
                picked.append(r)
    return sorted(picked, key=lambda r: -r.rows / r.n_rows)
