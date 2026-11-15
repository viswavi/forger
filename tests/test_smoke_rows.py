"""Tests for smoke runs: ``@pipeline(smoke=N)`` and ``FORGER_SMOKE_ROWS``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from fixtures import MockModel

from pyforger import _tracking, alter, filter, op
from pyforger.acquire import Model, retrieve_from_s2, retrieve_from_web, sample_from_LLM
from pyforger.op import spread_indices
from pyforger.steps import pipeline, render_pipeline, render_step, step

BANNER = (
    "LIMITED RUN (limit=3): at most 3 rows per step; outputs are not representative"
)


@pytest.fixture(autouse=True)
def _quiet_and_fresh(monkeypatch):
    """Silence reports, forget step records, and start without the variable."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    monkeypatch.delenv("FORGER_SMOKE_ROWS", raising=False)
    _tracking.clear_step_records()
    yield
    _tracking.clear_step_records()


def _rows(n: int, start: int = 0) -> list[dict]:
    return [{"id": i, "text": f"row number {i}"} for i in range(start, start + n)]


def _kept(rows: list[Any], limit: int) -> list[Any]:
    """The rows a limited run keeps of `rows`."""
    return [rows[i] for i in spread_indices(rows, limit)]


def _ids(rows: list[dict], limit: int) -> list[int]:
    return [r["id"] for r in _kept(rows, limit)]


SEEN: list[int] = []  # rows each _tag call got


@op(name="alter.test_smoke_tag")
def _tag(rows: list[Any]) -> list[Any]:
    # Through globals(), so the growing counter is not part of the steps' keys.
    globals()["SEEN"].append(len(rows))
    return [{**r, "tag": "t"} for r in rows]


def _keep_all(rows: list[Any]) -> Any:
    return filter.by_length(rows, instruction_field="text", min_len=1)


@step
def tag(rows: list[dict]) -> list[dict]:
    """Tag every row."""
    return _keep_all(_tag(rows))


@step
def merge(left: list[dict], *more: list[dict], extra: list[dict]) -> list[dict]:
    """Tag rows from several inputs."""
    return _tag(left + [r for rows in more for r in rows] + extra)


def _last() -> _tracking.PipelineRecord:
    rec = _tracking.last_pipeline()
    assert rec is not None
    return rec


def test_step_inputs_are_cut_to_rows_spread_over_them():
    """Each rows argument is cut, also *rows items and keyword rows."""
    SEEN.clear()

    @pipeline(smoke=3)
    def build() -> list[dict]:
        rows = tag(_rows(10))
        return merge(rows, _rows(5, 10), _rows(1, 20), extra=_rows(4, 30))

    out = build()
    assert SEEN == [3, 3 + 3 + 1 + 3]
    expected = _ids(_rows(10), 3) + _ids(_rows(5, 10), 3) + [20] + _ids(_rows(4, 30), 3)
    assert [r["id"] for r in out] == expected
    rec = _last()
    assert rec.smoke_rows == 3
    first, second = rec.steps
    assert (first.n_in, first.smoke_n_in, first.smoke_rows) == (3, 10, 3)
    assert [s.n_rows for s in second.inputs] == [3, 3, 1, 3]
    assert second.smoke_n_in == 3 + 5 + 1 + 4
    assert "LIMIT: 3 of 10 rows" in render_step(first)
    # The cut is not shown as rows dropped in plain Python between steps.
    assert rec.python_gaps == []


def test_banner_ends_the_summary_and_marks_step_reports(monkeypatch, capsys):
    """The summary's last line is the banner; step reports carry a SMOKE mark."""
    monkeypatch.delenv("FORGER_STEP_REPORT")

    @pipeline(smoke=3)
    def build() -> list[dict]:
        return tag(tag(_rows(2)))

    build()
    printed = capsys.readouterr().out
    assert printed.rstrip().splitlines()[-1].strip() == BANNER
    assert "LIMIT: ≤3 rows" in printed  # nothing to cut: 2 rows
    assert "LIMITED RUN" in render_pipeline(_last()).splitlines()[1]


@step
def picky(rows: list[dict]) -> list[dict]:
    """Keep rows with long text, expecting most of them."""
    return filter.by_length(rows, instruction_field="text", min_len=3, min_yield=0.9)


# One row in three has 3+ words: yield 0.33.
UNEVEN = [{"id": i, "text": "short" if i % 3 else "a longer text"} for i in range(9)]


def test_smoke_yield_miss_is_a_suggestion(monkeypatch, capsys):
    """On 3 rows a min_yield miss says little: a suggestion, not a data problem."""
    monkeypatch.delenv("FORGER_STEP_REPORT")

    @pipeline(smoke=3)
    def build() -> list[dict]:
        return picky(UNEVEN[:3])  # 3 rows: nothing to cut, so the yield is 0.33

    assert [r["id"] for r in build()] == [0]
    rec = _last()
    (only,) = rec.steps
    expected = "limited run: filter.by_length: yield 0.33 below min_yield 0.90"
    assert only.warnings == [] and only.suggestions == [expected]
    lines = [line.strip() for line in capsys.readouterr().out.splitlines()]
    lines = [line for line in lines if line]
    # No warnings, and the banner as the very last line.
    assert lines[-1] == BANNER
    assert not any(line.startswith("WARNING") for line in lines)
    assert f"INFO: picky: {expected}" in lines


def test_full_run_yield_miss_is_a_data_problem():
    """Outside a smoke run the same miss is a warning, the summary's last line."""

    @pipeline
    def build() -> list[dict]:
        return picky(UNEVEN)

    build()
    (only,) = _last().steps
    assert only.warnings == ["filter.by_length: yield 0.33 below min_yield 0.90"]
    assert not any("min_yield" in s for s in only.suggestions)
    summary = render_pipeline(_last())
    assert summary.rstrip().splitlines()[-1].strip() == (
        "WARNING: picky: filter.by_length: yield 0.33 below min_yield 0.90"
    )
    assert not summary.rstrip().endswith(BANNER)


def test_no_banner_in_a_full_run():
    """A pipeline without smoke has no banner and no step marks."""

    @pipeline
    def build() -> list[dict]:
        return tag(tag(_rows(5)))

    build()
    rec = _last()
    assert rec.smoke_rows is None
    assert "LIMIT" not in render_pipeline(rec)
    assert all(s.smoke_rows is None for s in rec.steps)


def test_limit_is_the_new_name_and_env_var(monkeypatch):
    """limit= and FORGER_LIMIT cut rows like smoke= and FORGER_SMOKE_ROWS."""

    @pipeline(limit=3)
    def build() -> list[dict]:
        return tag(_rows(10))

    assert len(build()) == 3
    monkeypatch.setenv("FORGER_LIMIT", "2")

    @pipeline
    def from_env() -> list[dict]:
        return tag(_rows(10))

    assert len(from_env()) == 2
    with pytest.raises(ValueError, match="not both"):
        pipeline(limit=3, smoke=3)


def test_limited_yield_miss_on_enough_rows_is_a_warning():
    """At limit=30 a min_yield miss is a data problem, as in a full run."""
    many = [{"id": i, "text": "short" if i % 3 else "a longer text"} for i in range(60)]

    @pipeline(limit=30)
    def build() -> list[dict]:
        return picky(many)

    build()
    (only,) = _last().steps
    share = sum(r["text"] == "a longer text" for r in _kept(many, 30)) / 30
    assert only.warnings == [
        f"filter.by_length: yield {share:.2f} below min_yield 0.90"
    ]


def test_limited_step_that_drops_every_row_warns():
    """A limited step returning nothing leaves later steps nothing to check."""

    @step
    def drop_all(rows: list[dict]) -> list[dict]:
        """Keep nothing."""
        return filter.by_length(rows, instruction_field="text", min_len=50)

    @pipeline(limit=3)
    def build() -> list[dict]:
        return tag(drop_all(_rows(10)))

    build()
    first = _last().steps[0]
    assert any("returned 0 of 3 rows in a limited run" in w for w in first.warnings)


def test_sample_from_llm_count_is_capped():
    """sample_from_LLM(n=100) asks for 3 samples in a smoke run."""
    model = MockModel([["a", "b", "c"]])

    @step
    def generate(topics: list[str], model: MockModel) -> list[dict]:
        """Sample from the model."""
        return sample_from_LLM("Write a riddle.", model, n=100)

    @pipeline(smoke=3)
    def build(model: MockModel) -> list[dict]:
        return tag(generate(["riddles"], model))

    assert [r["text"] for r in build(model)] == ["a", "b", "c"]
    assert model._call_count == 1


def test_retrieval_queries_and_count_are_capped():
    """3 of the 5 queries run, spread over the list, each with n capped to 3."""
    calls: list[tuple[Any, int]] = []

    def fetch(self, mode, *, query=None, n=10, seed=None, **kwargs):
        calls.append((query, n))
        return [{"text": f"{query} {i}"} for i in range(n)]

    @step
    def search(seeds: list[str]) -> list[dict]:
        """Search for five fixed topics."""
        return retrieve_from_web(queries=["a", "b", "c", "d", "e"], n=10)

    @pipeline(smoke=3)
    def build() -> list[dict]:
        return tag(search(["web"]))

    with patch("pyforger.acquire.sources.WebRetrievalSource.fetch", fetch):
        out = build()
    assert calls == [(q, 3) for q in _kept(["a", "b", "c", "d", "e"], 3)]
    assert len(out) == 3  # the next step's input is cut too


def test_concurrent_s2_queries_and_count_are_capped():
    """retrieve_from_s2 with max_workers > 1 is capped like retrieve."""
    calls: list[tuple[Any, int]] = []

    def fetch(self, mode, *, query=None, n=10, seed=None, **kwargs):
        calls.append((query, n))
        return [{"text": f"{query} {i}"} for i in range(n)]

    @step
    def search(seeds: list[str]) -> list[dict]:
        """Search for five fixed topics, two at a time."""
        return retrieve_from_s2(queries=["a", "b", "c", "d", "e"], n=10, max_workers=2)

    @pipeline(smoke=3)
    def build() -> list[dict]:
        return tag(search(["papers"]))

    with patch("pyforger.acquire.sources.S2RetrievalSource.fetch", fetch), patch(
        "pyforger.acquire.sources.S2RetrievalSource._get_client"
    ):
        build()
    assert sorted(calls) == [(q, 3) for q in _kept(["a", "b", "c", "d", "e"], 3)]


def test_environment_variable_and_decorator_forms(monkeypatch):
    """FORGER_SMOKE_ROWS turns it on; smoke=0 turns it off; bad values raise."""
    SEEN.clear()

    @pipeline
    def from_env() -> list[dict]:
        return tag(tag(_rows(10)))

    @pipeline(smoke=0)
    def never() -> list[dict]:
        return tag(tag(_rows(10, 100)))

    monkeypatch.setenv("FORGER_SMOKE_ROWS", "2")
    from_env()
    assert _last().smoke_rows == 2 and SEEN == [2, 2]
    never()
    assert _last().smoke_rows is None and SEEN[2:] == [10, 10]

    monkeypatch.setenv("FORGER_SMOKE_ROWS", "0")
    from_env()
    assert _last().smoke_rows is None

    for bad in ("three", "-1"):
        monkeypatch.setenv("FORGER_SMOKE_ROWS", bad)
        with pytest.raises(ValueError, match="FORGER_SMOKE_ROWS"):
            from_env()
    with pytest.raises(ValueError, match="smoke"):
        pipeline(smoke=-2)


def test_full_run_after_a_smoke_run_recomputes_every_step():
    """Smoke results are cached apart: neither run reads the other's entries."""
    SEEN.clear()

    def build_with(smoke: int | None) -> Any:
        @pipeline(name="build", smoke=smoke)
        def build() -> list[dict]:
            return tag(tag(_rows(3)))  # 3 rows: the smoke run cuts nothing

        return build()

    build_with(3)
    assert SEEN == [3, 3]
    build_with(None)
    full = _last()
    assert not any(s.cache_hit for s in full.steps)
    assert SEEN == [3, 3, 3, 3]

    build_with(3)  # the smoke run reuses its own entries
    assert all(s.cache_hit for s in _last().steps)
    build_with(None)
    assert all(s.cache_hit for s in _last().steps)
    assert SEEN == [3, 3, 3, 3]


def test_ops_outside_a_smoke_pipeline_are_untouched():
    """Without a running pipeline nothing is capped or cut."""
    model = MockModel([["a"] * 10])
    rows = sample_from_LLM("Write a riddle.", model, n=10)
    assert len(rows) == 10
    assert _tracking.smoke_rows() is None


@pytest.fixture
def sent() -> Any:
    """Patch LiteLLM with a fake that records each request as (prompt, n).

    The j-th completion of the k-th request for "Q" is "Q#k.j", so a reused
    completion can be told from a new one.
    """
    calls: list[tuple[str, int]] = []

    def completion(messages, n=1, **kwargs):
        text = messages[-1]["content"]
        calls.append((text, n))
        k = len(calls)
        choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content=f"{text}#{k}.{j}", reasoning_content=None
                ),
                finish_reason="stop",
            )
            for j in range(n)
        ]
        return SimpleNamespace(
            choices=choices,
            usage=SimpleNamespace(
                prompt_tokens=1, completion_tokens=n, completion_tokens_details=None
            ),
            _hidden_params={"response_cost": 0.0},
        )

    with patch("pyforger.acquire.model.litellm") as litellm:
        litellm.get_supported_openai_params.return_value = None
        litellm.completion.side_effect = completion
        yield calls


@step
def ask(rows: list[dict], model: Model) -> list[dict]:
    """Ask the model about each row."""
    return _keep_all(alter.generate_from_context(rows, "Say {text}", model))


def test_full_run_after_a_smoke_run_sends_only_new_rows(sent):
    """The request cache ignores the smoke limit, so smoke rows are not re-asked."""
    model = Model("gpt-4o", show_progress=False)

    def build_with(smoke: int | None) -> Any:
        @pipeline(name="build", smoke=smoke)
        def build(model: Model) -> list[dict]:
            return ask(_rows(5), model)

        return build(model)

    smoke = build_with(2)
    asked = _ids(_rows(5), 2)
    assert sorted(sent) == [(f"Say row number {i}", 1) for i in asked]
    sent.clear()
    full = build_with(None)
    assert not any(s.cache_hit for s in _last().steps)  # a new step entry
    rest = [i for i in range(5) if i not in asked]
    assert sorted(sent) == [(f"Say row number {i}", 1) for i in rest]
    assert [full[i] for i in asked] == smoke  # the smoke run's completions, reused


def test_capped_sample_count_keeps_its_samples_for_the_full_run(sent):
    """With n outside the request scope, n=5 after a smoke n=2 asks for 3 more."""
    model = Model("gpt-4o", show_progress=False)

    @step
    def generate(topics: list[str], model: Model) -> list[dict]:
        """Sample from the model."""
        return sample_from_LLM("Write a riddle.", model, n=5)

    def build_with(smoke: int | None) -> Any:
        @pipeline(name="build", smoke=smoke)
        def build(model: Model) -> list[dict]:
            return tag(generate(["riddles"], model))

        return build(model)

    smoke = build_with(2)
    assert sent == [("Write a riddle.", 2)]
    sent.clear()
    full = build_with(None)
    assert sent == [("Write a riddle.", 3)]  # only the missing slots
    assert [r["text"] for r in full[:2]] == [r["text"] for r in smoke]


@op(name="alter.test_smoke_outer")
def _outer(rows: list[Any]) -> list[Any]:
    # An op that expands its rows and hands them to another op.
    doubled = [{**r, "copy": c} for r in rows for c in (0, 1)]
    return _tag(doubled)


@step
def build_from_nothing(rows: list[dict]) -> list[dict]:
    """Make many prompts in Python from a few input rows, then tag them."""
    made = [{"id": 100 + i, "text": f"made {i}"} for i in range(40)]
    return _tag(made)


@step
def build_by_keyword(rows: list[dict]) -> list[dict]:
    """Pass the built rows by keyword."""
    return _tag(rows=[{"id": 200 + i, "text": f"kw {i}"} for i in range(12)])


@step
def expand(rows: list[dict]) -> list[dict]:
    """An op that expands rows inside itself keeps them all."""
    return _outer(rows)


def test_rows_built_in_a_step_are_cut_too(caplog):
    """Rows a step builds in Python reach its ops cut to the limit, with a log line."""
    SEEN.clear()

    @pipeline(limit=3)
    def build() -> list[dict]:
        rows = build_from_nothing(_rows(10))
        return build_by_keyword(rows)

    with caplog.at_level("INFO", logger="forger.op"):
        out = build()
    assert SEEN == [3, 3]
    built = [{"id": 200 + i, "text": f"kw {i}"} for i in range(12)]
    assert [r["id"] for r in out] == _ids(built, 3)
    lines = [r.getMessage() for r in caplog.records if "limited run" in r.getMessage()]
    assert lines[0] == (
        "alter.test_smoke_tag: limited run: using 3 of its 40 input rows, spread over "
        "the list; a full run sends all of them, so scale this op's cost by 40 / 3"
    )


def test_merged_step_inputs_and_nested_ops_are_not_cut():
    """A step's own merged inputs stay whole, and an op inside an op is not cut."""
    SEEN.clear()

    @pipeline(limit=3)
    def build() -> list[dict]:
        merge(_rows(5), _rows(5, 10), extra=_rows(5, 30))  # 3 + 3 + 3 rows
        return expand(_rows(10))

    out = build()
    assert SEEN == [9, 6]  # merge keeps 9; _outer's 6 doubled rows reach _tag
    assert len(out) == 6


def test_full_run_ops_are_not_cut():
    """Without a limit, rows built in a step reach the op whole."""
    SEEN.clear()

    @pipeline
    def build() -> list[dict]:
        return build_from_nothing(_rows(2))

    assert len(build()) == 40 and SEEN == [40]


def test_spread_indices_take_one_row_per_block():
    """3 of 50 come from the first, middle and last third, in list order."""
    rows = [{"query": f"q{i}"} for i in range(50)]
    picks = spread_indices(rows, 3)
    assert [0 <= picks[0] < 16, 16 <= picks[1] < 33, 33 <= picks[2] < 50] == [True] * 3
    assert picks == spread_indices(
        [dict(r) for r in rows], 3
    )  # by content, not identity
    assert spread_indices(rows[:2], 3) == [0, 1]


def test_rows_written_per_seed_come_from_many_seeds():
    """12 of 36 chats written 3 per seed come from 12 seeds, not the first 4."""
    chats = [{"seed": s, "k": k} for s in range(12) for k in range(3)]
    assert sorted({chats[i]["seed"] for i in spread_indices(chats, 12)}) == list(
        range(12)
    )


@op(name="alter.test_smoke_pairs", category="alter")
def _pairs(inputs: list[str], candidates: list[str]) -> list[dict]:
    assert len(inputs) == len(candidates)
    return [{"text": i, "candidate": c} for i, c in zip(inputs, candidates)]


@step
def pair_up(rows: list[dict]) -> list[dict]:
    """Build two aligned lists in Python and pass both to one op."""
    texts = [f"t{i}" for i in range(20)]
    return _pairs(texts, [t.upper() for t in texts])


def test_aligned_list_arguments_are_cut_at_the_same_positions():
    """An op's other list arguments of the same length stay aligned with its rows."""

    @pipeline(limit=3)
    def build() -> list[dict]:
        return pair_up(_rows(1))

    out = build()
    assert len(out) == 3
    assert all(r["candidate"] == r["text"].upper() for r in out)
