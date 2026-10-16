"""Tests for scripts/gen_op_docs.py and the generated docs/OPS.md."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from pyforger.op import op, tracked

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gen_op_docs.py"
_spec = importlib.util.spec_from_file_location("gen_op_docs", _SCRIPT)
assert _spec is not None and _spec.loader is not None
gen_op_docs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_op_docs)

_PRIVATE_DEFAULT = "v2"


def test_ops_md_is_up_to_date():
    """docs/OPS.md matches what the generator renders from the current code."""
    current = gen_op_docs.OUTPUT.read_text()
    assert current == gen_op_docs.render(), (
        "docs/OPS.md is out of date with the ops' signatures or docstrings; "
        "run python scripts/gen_op_docs.py"
    )


@op(name="filter.by_fake")
def _fake_filter(
    items: list[dict],
    *,
    field: str,
    threshold: float = 0.5,
    version: str = _PRIVATE_DEFAULT,
) -> list[dict]:
    """Keep items whose `field` is long enough.

    Second paragraph, not rendered.

    Args:
        items: Rows to filter; not rendered.

    Returns:
        Kept items, each with a ``score``
            field added.

        A second paragraph.

    Raises:
        ValueError: Not rendered.
    """
    return items


def test_signature_unwraps_op_and_hides_its_keywords():
    """The @op wrapper's _no_cache / min_yield keywords are not in the signature."""
    sig = gen_op_docs.render_signature("by_fake", _fake_filter)
    assert sig == (
        "by_fake(items: list[dict], *, field: str, threshold: float = 0.5, "
        'version: str = "v2") -> list[dict]'
    )
    assert "min_yield" not in sig and "_no_cache" not in sig


def test_long_signature_wraps_one_parameter_per_line():
    """Signatures over 120 characters put each parameter on its own line."""

    def long_op(
        rows: list[dict],
        *,
        a_very_long_parameter_name: str = "x" * 40,
        another_long_parameter_name: int | None = None,
        **kw: Any,
    ) -> list[dict]:
        """Do nothing."""
        return rows

    sig = gen_op_docs.render_signature("long_op", long_op)
    assert sig.splitlines() == [
        "long_op(",
        "    rows: list[dict],",
        "    *,",
        '    a_very_long_parameter_name: str = "x" * 40,',
        "    another_long_parameter_name: int | None = None,",
        "    **kw: Any,",
        ") -> list[dict]",
    ]


def test_status_reports_caching():
    """Cached ops, uncached ops, tracked functions, and plain functions differ."""

    @op(name="alter.uncached", cache=False)
    def uncached(items):
        """Do nothing."""
        return items

    @tracked("acquire.fetch")
    def fetch(query):
        """Do nothing."""
        return []

    assert gen_op_docs._status(_fake_filter, "filter.by_fake") == (
        "Disk-cached (`@op`)."
    )
    assert "named `filter.by_fake`" in gen_op_docs._status(_fake_filter, "x.y")
    assert "caching off" in gen_op_docs._status(uncached, "alter.uncached")
    assert "Not cached (wrap it" in gen_op_docs._status(fetch, "acquire.fetch")
    assert gen_op_docs._status(len, "len") == "Not cached: a plain function."


def test_entry_renders_first_paragraph_and_returns_only():
    """An entry has the first paragraph and Returns:, not Args: or Raises:."""
    entry = gen_op_docs._entry("by_fake", "filter.by_fake", _fake_filter)
    assert entry.endswith(
        "Keep items whose `field` is long enough.\n\n"
        "**Returns:** Kept items, each with a `score` field added.\n\n"
        "A second paragraph.\n"
    )
    assert "Second paragraph" not in entry
    assert "Rows to filter" not in entry and "Not rendered" not in entry


def test_entry_without_returns_section():
    """A docstring without Returns: renders no Returns line."""

    def plain(rows):
        """Do nothing."""
        return rows

    assert gen_op_docs._returns_section(plain) is None
    assert "Returns" not in gen_op_docs._entry("plain", "alter.plain", plain)
