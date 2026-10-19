"""Tests for the pipeline check that retrieved text reaches a prompt or the output."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from pyforger import _tracking, alter, filter
from pyforger.acquire import Model, retrieve_from_web
from pyforger.steps import pipeline, render_pipeline, step


@pytest.fixture(autouse=True)
def _quiet_and_fresh(monkeypatch):
    """Silence reports and forget earlier step records."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    monkeypatch.delenv("FORGER_MAX_COST_USD", raising=False)
    _tracking.clear_step_records()
    yield
    _tracking.clear_step_records()


def _page(i: int) -> dict:
    words = " ".join(f"fact{i}x{j}" for j in range(60))
    return {"text": f"Page {i} says: {words}.", "id": f"http://p/{i}", "url": f"p{i}"}


@pytest.fixture
def pages() -> Iterator[list[dict]]:
    """Brave search returns three long pages; no network."""
    found = [_page(i) for i in range(3)]
    with patch(
        "pyforger.acquire.sources.WebRetrievalSource.fetch",
        side_effect=lambda *a, **k: [dict(p) for p in found],
    ):
        yield found


def _response(text: str = "an answer") -> SimpleNamespace:
    message = SimpleNamespace(content=text, reasoning_content=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            prompt_tokens=1, completion_tokens=1, completion_tokens_details=None
        ),
        _hidden_params={"response_cost": 0.0},
    )


@pytest.fixture
def model() -> Iterator[Model]:
    """A LiteLLM Model whose provider calls are faked; it records each prompt."""
    sent: list[Any] = []

    def completion(messages, **kwargs):
        sent.append(messages)
        return _response()

    def batch_completion(messages, **kwargs):
        sent.extend(messages)
        return [_response() for _ in messages]

    m = Model("openai/test-model")
    with patch("pyforger.acquire.model.litellm") as litellm:
        litellm.completion.side_effect = completion
        litellm.batch_completion.side_effect = batch_completion
        m.sent = sent  # type: ignore[attr-defined]
        yield m


@step
def search(topics: list[dict]) -> list[dict]:
    """Search the web for the first topic."""
    return retrieve_from_web(queries=[t["topic"] for t in topics[:1]], n=3)


def _keep(rows: list[dict]) -> list[dict]:
    return filter.by_length(rows, instruction_field="text", min_len=0)


def _run(prompt: Any, model: Model, *, keep_pages: bool = False) -> Any:
    @step
    def write(docs: list[dict], model: Model) -> list[dict]:
        rows = alter.generate_from_context(docs, prompt, model)
        return _keep([r for r in rows if r])

    @pipeline
    def build(model: Model) -> list[dict]:
        docs = search([{"topic": "tides"}])
        out = write(docs, model)
        return out + docs if keep_pages else out

    build(model)
    rec = _tracking.last_pipeline()
    assert rec is not None
    return rec


def _reach_warnings(rec: Any) -> list[str]:
    return [w for w in rec.warnings if "retrieve_from_web" in w]


def test_unused_retrieval_warns(pages, model):
    """Pages kept in rows whose prompt formats only the query give the warning."""
    rec = _run("Write a question about {query}.", model)
    assert len(model.sent) == 3
    assert _reach_warnings(rec) == [
        "acquire.retrieve_from_web returned 3 documents, but none of their text "
        "appeared in any of the 3 model prompts or in the pipeline's output; to "
        "ground generation on them, format the retrieved text into your prompts"
    ]
    assert "format the retrieved text" in render_pipeline(rec)


def test_text_in_a_prompt_counts_as_used(pages, model):
    """A prompt that formats the page text gives no warning and no note."""
    rec = _run("Ask about this page:\n{text}", model)
    assert _reach_warnings(rec) == []
    assert rec.notes == []


def test_reformatted_text_in_a_chat_prompt_counts(pages, model):
    """Chat-message content parts count, with whitespace changed."""

    def prompt(row: dict) -> list[dict]:
        wrapped = "\n".join(row["text"].split(" "))  # one word per line
        return [{"role": "user", "content": [{"type": "text", "text": wrapped}]}]

    rec = _run(prompt, model)
    assert _reach_warnings(rec) == []


def test_text_only_in_the_returned_rows_counts_as_used(pages, model):
    """Pages the pipeline returns as data count as used."""
    rec = _run("Write a question about {query}.", model, keep_pages=True)
    assert _reach_warnings(rec) == []


def test_few_documents_used_is_a_note(model):
    """One page of 30 in a prompt is a note, not a warning."""
    many = [_page(i) for i in range(30)]
    with patch(
        "pyforger.acquire.sources.WebRetrievalSource.fetch",
        side_effect=lambda *a, **k: [dict(p) for p in many],
    ):
        rec = _run(
            lambda r: f"Ask: {r['text'] if r['id'] == 'http://p/0' else ''}", model
        )
    assert _reach_warnings(rec) == []
    assert rec.notes == [
        "1 of 30 documents from acquire.retrieve_from_web (3%) appeared in a model "
        "prompt or the pipeline's output"
    ]
    assert "notes" in render_pipeline(rec)


def test_no_warning_without_retrieval(model):
    """A pipeline that retrieves nothing gets no retrieval warning."""

    @step
    def write(rows: list[dict], model: Model) -> list[dict]:
        return _keep(alter.generate_from_context(rows, "Say {topic}.", model))

    @pipeline
    def build(model: Model) -> list[dict]:
        return write([{"topic": "tides"}], model)

    build(model)
    rec = _tracking.last_pipeline()
    assert rec is not None and not any("documents" in w for w in rec.warnings)


def test_cached_rerun_gives_the_same_verdict(pages, model):
    """Cached steps replay their documents and prompts: same verdict, no requests."""
    first = _run("Write a question about {query}.", model)
    assert len(model.sent) == 3
    again = _run("Write a question about {query}.", model)
    assert len(model.sent) == 3  # every step came from the cache
    assert all(s.cache_hit for s in again.steps)
    assert _reach_warnings(again) == _reach_warnings(first) != []

    _run("Ask about this page:\n{text}", model)
    sent = len(model.sent)
    used = _run("Ask about this page:\n{text}", model)
    assert len(model.sent) == sent
    assert _reach_warnings(used) == []


def test_cache_hit_without_a_record_stays_silent(pages, model, tmp_path):
    """A result cached before this check existed has unknown prompts: no warning."""
    first = _run("Write a question about {query}.", model)
    assert _reach_warnings(first) != []
    records = list(tmp_path.rglob("*.reach.json"))
    assert records
    for record in records:
        record.unlink()  # as if cached by an older Forger
    again = _run("Write a question about {query}.", model)
    assert all(s.cache_hit for s in again.steps)
    assert _reach_warnings(again) == [] and again.notes == []


def test_op_cache_hit_in_a_changed_step_replays_its_prompts(pages, model):
    """A new step whose generate op is cached still counts that op's prompts."""
    _run("Ask about this page:\n{text}", model)
    sent = len(model.sent)

    @step
    def write_v2(docs: list[dict], model: Model) -> list[dict]:
        rows = alter.generate_from_context(docs, "Ask about this page:\n{text}", model)
        return _keep([r for r in rows if r])  # same op calls, new step

    @pipeline
    def build(model: Model) -> list[dict]:
        return write_v2(search([{"topic": "tides"}]), model)

    build(model)
    rec = _tracking.last_pipeline()
    assert rec is not None
    assert not rec.steps[1].cache_hit and rec.steps[1].ops[0].cache_hit
    assert len(model.sent) == sent  # no request was sent
    assert _reach_warnings(rec) == []


def test_no_tracking_outside_a_pipeline(pages, model, tmp_path):
    """Outside a pipeline nothing is tracked or cached, and nothing fails."""
    docs = retrieve_from_web(queries=["tides"], n=3)
    rows = alter.generate_from_context(docs, "Write about {query}.", model)
    assert len(rows) == 3
    assert not list(tmp_path.rglob("*.reach.json"))
    with _tracking.reach_scope() as log:
        assert log is None
    _tracking.note_prompt_texts(["anything"])  # a no-op, not an error
    _tracking.register_documents("acquire.retrieve", docs)


def test_fingerprints_skip_short_documents_and_cap_the_count():
    """Short texts get no fingerprints; at most 2,000 documents are fingerprinted."""
    assert _tracking.fingerprints("too short") == []
    fps = _tracking.fingerprints(_page(0)["text"])
    assert len(fps) == 3 and all(len(fp) == 60 for fp in fps)
    assert all(fp in " ".join(_page(0)["text"].split()) for fp in fps)

    rec = _tracking.PipelineRecord(name="p")
    scope = _tracking.enter_pipeline(rec)
    try:
        rows = [_page(i) for i in range(_tracking._MAX_DOCS + 50)]
        _tracking.register_documents("acquire.retrieve", rows)
        reach = _tracking._running_for_context().reach  # type: ignore[union-attr]
        assert len(reach.docs) == _tracking._MAX_DOCS
        assert reach.returned == {"acquire.retrieve": _tracking._MAX_DOCS + 50}
        _tracking.note_prompt_texts([rows[0]["text"]])
        assert reach.used == {0}
        assert all(index != 0 for index in reach.pending.values())
    finally:
        _tracking.exit_pipeline(scope)
