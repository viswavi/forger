"""Tests for summaries of long sample values in @step reports."""

from __future__ import annotations

import io
import json
import threading

import pytest
from rich.console import Console

from pyforger import _summaries, _tracking, op
from pyforger._render import render_step, sample_summaries, step_renderable
from pyforger._tracking import StepRecord
from pyforger.acquire import model as model_module
from pyforger.inspect import inspect_step
from pyforger.op import trace
from pyforger.steps import pipeline, set_report_model, step

CODE = (
    "import sys\n\n\ndef solve(n: int) -> int:\n    seen = set()\n"
    "    for i in range(n):\n        seen.add(i * i % 7)\n    return len(seen)\n\n"
    + "".join(f"def helper_{i}(x):\n    return x + {i}\n\n" for i in range(100))
)
PROSE = (
    "The patient is a 54-year-old man with three days of chest pain. " * 40
    + "He was discharged on aspirin with cardiology follow-up."
)
FILES = ["src/a.py", "src/b.py", "src/c.py", "tests/test_a.py", "README.md"] * 8
RECORD = {
    "name": "Ada Lovelace",
    "files": FILES[:5],
    "meta": {"year": 1843, "notes": "n" * 300},
    "score": 0.7,
}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    """Quiet reports, no report model, and no step records left from other tests."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    monkeypatch.delenv("FORGER_REPORT_MODEL", raising=False)
    _tracking.clear_step_records()
    _summaries._ENV_MODELS.clear()
    yield
    _summaries._REPORT_MODEL[0] = _summaries._UNSET
    _summaries._ENV_MODELS.clear()
    _tracking.clear_step_records()


class _FakeSummarizer:
    """Model stand-in: counts calls, records prompts, and reports some usage."""

    def __init__(
        self, reply: str = "A Python solver that counts squares mod 7."
    ) -> None:
        self.reply = reply
        self.prompts: list[str] = []
        self.fail = False
        self._lock = threading.Lock()

    @property
    def calls(self) -> int:
        return len(self.prompts)

    def cache_key(self) -> tuple:
        return ("_FakeSummarizer", self.reply)

    def generate(self, prompt, *, n=1, temperature=None, max_tokens=None):
        with self._lock:
            self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("provider down")
        _tracking.add_usage(1000, 50, 0.5)
        return [f'"{self.reply}"\nextra line'] * n


@op(name="alter.test_identity")
def _identity(rows: list[dict]) -> list[dict]:
    return list(rows)


@op(name="filter.test_keep_all")
def _keep_all(rows: list[dict]) -> list[dict]:
    return _tracking.keep_where(rows, [True] * len(rows))


def _make_step(**kwargs):
    @step(**kwargs)
    def write_code(rows):
        """Write a solver for each problem."""
        return _keep_all(_identity(rows))

    return write_code


def _compact(value) -> str | None:
    return _summaries.compact_summary(value, 80)


def _record(row: dict) -> StepRecord:
    return StepRecord(
        name="demo",
        n_in=1,
        n_out=1,
        sample=[row],
        sample_summaries=[sample_summaries(row)],
    )


# --------------------------------------------------------------------------- #
# Structural summaries
# --------------------------------------------------------------------------- #


def test_code_shows_first_meaningful_lines_and_size():
    """Code skips imports and blank lines, then gives lines, chars, and language."""
    text = render_step(_record({"solution": CODE}))
    lines = CODE.count("\n") + 1
    assert "def solve(n: int) -> int:" in text
    assert f"({lines} lines, {len(CODE) / 1e3:.1f}k chars, python)" in text
    assert "import sys" not in text


def test_list_dict_and_json_string_are_summarized_as_structures():
    """Lists give a count and items, dicts keys with short value previews."""
    files, record = _compact(FILES), _compact(RECORD)
    assert files == (
        "[40 items] src/a.py, src/b.py, src/c.py, tests/test_a.py, README.md, "
        "src/a.py, …"
    )
    assert record == (
        "{4 keys} name: Ada Lovelace, files: [5 items], meta: {2 keys}, score: 0.7"
    )
    assert _compact(json.dumps(RECORD)).startswith("JSON {4 keys} name: Ada Lovelace")
    assert all(len(x) <= 80 for x in (files, record))
    text = render_step(_record({"files": FILES, "record": RECORD}))
    assert "[40 items] src/a.py" in text and "{4 keys} name: Ada" in text


def test_prose_shows_head_and_tail_with_total_length():
    """Long prose keeps its start and end and says how long it is."""
    text = render_step(_record({"note": PROSE}))
    words = len(PROSE.split())
    assert "The patient is a 54-year-old man" in text
    assert "… follow-up." in text
    assert f"({words:,} words, {len(PROSE) / 1e3:.1f}k chars)" in text


def test_short_values_are_unchanged():
    """Values that fit are shown as before, with no summary attached."""
    row = {"text": "short answer", "n": 3, "tags": ["a", "b"]}
    assert sample_summaries(row) == {}
    text = render_step(_record(row))
    assert "short answer" in text and '["a", "b"]' in text and "items" not in text


def test_full_view_shows_longer_previews():
    """inspect_step's view keeps code lines and lists one item per line."""
    text = render_step(_record({"solution": CODE, "files": FILES}), full=True)
    assert "    for i in range(n):" in text
    assert "\n" in text and "- tests/test_a.py" in text and "… 20 more" in text
    assert "python)" in text


def test_summaries_use_the_whole_value_not_the_stored_sample():
    """A step's sample is cut to 500 chars, but the summary counts every line."""
    out = _make_step()([{"solution": CODE}])
    rec = _tracking.step_records()["write_code"]
    assert out[0]["solution"] == CODE
    assert len(rec.sample[0]["solution"]) == 500
    assert f"({CODE.count(chr(10)) + 1} lines" in render_step(rec)


def test_hostile_long_values_never_reach_the_terminal_raw():
    """Escapes in long values are neutralized, and narrow consoles still fit."""
    bad = "\x1b[31mRED\x1b[0m\ttab\x07"
    row = {
        "code": f"def f():\n    return '{bad}'\n" * 30,
        "list": [bad] * 30,
        "dict": {bad: bad * 20, "b": [bad] * 9},
        "json": json.dumps({bad: [bad] * 20}),
        "prose": (bad + " words ") * 80,
    }
    rec = _record(row)
    for full in (False, True):
        text = render_step(rec, full=full)
        assert "\x1b" not in text and "\x07" not in text and "\t" not in text
        assert all(len(line) <= 120 for line in text.splitlines())
    for width in (40, 80):
        console = Console(file=io.StringIO(), width=width, color_system=None)
        console.print(step_renderable(rec, width=width))
        assert all(len(line) <= width for line in console.file.getvalue().splitlines())


def test_odd_values_fall_back_to_the_cut_view():
    """A long value with no shape (a number, an object) is cut as before."""
    row = {"big": 10**200, "obj": object()}
    assert sample_summaries(row) == {}
    text = render_step(_record(row))
    assert "…" in text


# --------------------------------------------------------------------------- #
# LLM summaries
# --------------------------------------------------------------------------- #


def _run_pipeline(rows, **step_kwargs):
    write_code = _make_step(**step_kwargs)

    @step
    def keep(rows):
        return _keep_all(rows)

    @pipeline
    def main(rows):
        return keep(write_code(rows))

    return main(rows)


def test_llm_summaries_are_opt_in_and_context_sensitive(monkeypatch, capsys):
    """Only long values get a summary; the prompt names step, docstring, field."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "1")
    fake = _FakeSummarizer()
    set_report_model(fake)
    _run_pipeline([{"solution": CODE, "files": FILES, "id": 7}])
    out = capsys.readouterr().out
    # Two long values in each of the two steps; "id" is short.
    assert fake.calls == 4
    by_field = {
        p.split("Field: ")[1].split("\n")[0]: p
        for p in fake.prompts
        if 'step "write_code"' in p
    }
    assert set(by_field) == {"solution", "files"}
    prompt = by_field["solution"]
    assert 'step "write_code"' in prompt
    assert "Write a solver for each problem." in prompt
    assert "def solve(n: int) -> int:" in prompt
    assert "A Python solver that counts squares mod 7. · " in out
    assert "extra line" not in out


def test_llm_summaries_stay_out_of_step_and_pipeline_records(monkeypatch):
    """Summary calls are not ops of any step and add no tokens or cost."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "1")
    fake = _FakeSummarizer()
    set_report_model(fake)
    _run_pipeline([{"solution": CODE}], strict=True)
    assert fake.calls == 2  # one per step: the prompt names the step
    pipe = _tracking.last_pipeline()
    assert pipe is not None
    assert [s.name for s in pipe.steps] == ["write_code", "keep"]
    assert not any("outside a @step" in w for w in pipe.warnings)
    assert (pipe.prompt_tokens, pipe.completion_tokens, pipe.cost_usd) == (0, 0, 0.0)
    for rec in pipe.steps:
        assert all(call.op_name != "report.summarize" for call in rec.ops)
        assert (rec.prompt_tokens, rec.cost_usd) == (0, 0.0)
    assert "report.summarize" in [r.op_name for r in trace()]


def test_summaries_count_only_toward_the_process_total(monkeypatch):
    """Summary calls are real spend: in the process total, not in the step's."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "1")
    fake = _FakeSummarizer()
    set_report_model(fake)
    before = _tracking.usage_snapshot()
    _make_step()([{"solution": CODE}])
    assert fake.calls == 1
    after = _tracking.usage_snapshot()
    assert (after[0] - before[0], after[1] - before[1]) == (1000, 50)
    rec = list(_tracking.step_records().values())[-1]
    assert (rec.prompt_tokens, rec.cost_usd) == (0, 0.0)


def test_summary_ops_do_not_count_as_started_ops():
    """A step running in another thread must not mistake a summary for its op."""
    fake = _FakeSummarizer()
    before = _tracking.ops_started()
    _tracking.run_uncounted(_summaries._summarize_one, fake, "prompt")
    assert fake.calls == 1
    assert _tracking.ops_started() == before


def test_reasoning_before_the_summary_is_dropped():
    """An inline <think> block is not mistaken for the summary sentence."""
    fake = _FakeSummarizer(reply="<think>\nLet me look.\n</think>\nA list of files.")
    assert _summaries._summarize_one(fake, "prompt") == "A list of files."


def test_shebang_without_interpreter_is_still_summarized():
    """A '#!' line with no interpreter name does not break the code summary."""
    code = "#!/usr/bin/ \n" + "def f(x):\n    return x\n" * 30
    assert _compact(code).endswith("chars, shell)")
    assert _compact("#!/usr/bin/env python3\n" + code[13:]).endswith("python3)")


def test_summaries_are_cached_on_disk(monkeypatch):
    """Printing again, inspecting, or rerunning the step makes no new calls."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "1")
    fake = _FakeSummarizer()
    set_report_model(fake)
    write_code = _make_step(cache=False)
    write_code([{"solution": CODE}])
    assert fake.calls == 1
    rec = inspect_step("write_code")
    assert fake.calls == 1
    assert (
        rec.sample_summaries[0][0].llm == "A Python solver that counts squares mod 7."
    )
    write_code([{"solution": CODE}])  # a new record; same value and context
    assert fake.calls == 1


def test_inspect_step_summarizes_rows_the_compact_report_skipped(monkeypatch):
    """The step report summarizes its shown rows; inspect_step the rest."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "1")
    fake = _FakeSummarizer()
    set_report_model(fake)
    rows = [{"solution": CODE + f"\n# variant {i}\n"} for i in range(5)]
    _make_step(sample=1)(rows)
    assert fake.calls == 1
    inspect_step("write_code")
    assert fake.calls == 5


def test_summary_calls_per_report_are_bounded(monkeypatch):
    """At most 8 values are summarized per printed report."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "1")
    fake = _FakeSummarizer()
    set_report_model(fake)
    row = {f"f{i}": CODE + f"\n# field {i}\n" for i in range(12)}
    _make_step()([row])
    assert fake.calls == _summaries._MAX_SUMMARY_CALLS == 8


def test_failed_summaries_fall_back_to_structure(monkeypatch, capsys):
    """A failing model never breaks the report; the structural summary shows."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "1")
    fake = _FakeSummarizer()
    fake.fail = True
    set_report_model(fake)
    out = _make_step()([{"solution": CODE}])
    assert out == [{"solution": CODE}]
    printed = capsys.readouterr().out
    assert "def solve(n: int) -> int:" in printed and "python)" in printed
    rec = _tracking.step_records()["write_code"]
    assert rec.sample_summaries[0][0].llm == ""
    inspect_step("write_code")  # a failed summary is not retried
    assert fake.calls == 1


def test_stream_samples_get_llm_summaries(monkeypatch, capsys):
    """Each stream of a dict result has its sample values summarized too."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "1")
    fake = _FakeSummarizer()
    set_report_model(fake)

    @step
    def split(rows):
        rows = _keep_all(_identity(rows))
        return {"train": rows, "held_out": [{"solution": CODE + "\n# held out"}]}

    split([{"solution": CODE}])
    assert fake.calls == 2  # one long value in each stream's sample row
    out = capsys.readouterr().out
    assert out.count("A Python solver that counts squares mod 7. · ") == 2
    rec = _tracking.step_records()["split"]
    assert rec.streams is not None
    assert all(s.sample_summaries[0][0].llm for s in rec.streams)


def test_silenced_reports_make_no_calls(monkeypatch):
    """FORGER_STEP_REPORT=0 and prod=True make no summary calls."""
    fake = _FakeSummarizer()
    set_report_model(fake)
    _make_step()([{"solution": CODE}])
    inspect_step("write_code")
    monkeypatch.setenv("FORGER_STEP_REPORT", "1")
    _make_step(prod=True)([{"solution": CODE}])
    assert fake.calls == 0


def test_env_var_builds_the_report_model(monkeypatch):
    """FORGER_REPORT_MODEL names a LiteLLM model; the key comes from the env."""
    built: list[tuple[str, dict]] = []
    fake = _FakeSummarizer()

    def fake_model(model_id, **kwargs):
        built.append((model_id, kwargs))
        return fake

    monkeypatch.setattr(model_module, "Model", fake_model)
    monkeypatch.setenv("FORGER_REPORT_MODEL", "litellm_proxy/test-model")
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")
    monkeypatch.setenv("FORGER_STEP_REPORT", "1")
    _make_step()([{"solution": CODE}])
    _make_step()([{"solution": CODE + "\n# again\n"}])
    assert [b[0] for b in built] == ["litellm_proxy/test-model"]  # built once
    assert built[0][1]["api_key"] == "test-key"
    assert fake.calls == 2


def test_set_report_model_none_overrides_the_env(monkeypatch):
    """set_report_model(None) turns summaries off even with FORGER_REPORT_MODEL."""
    monkeypatch.setattr(
        model_module, "Model", lambda *a, **k: pytest.fail("Model was built")
    )
    monkeypatch.setenv("FORGER_REPORT_MODEL", "litellm_proxy/test-model")
    monkeypatch.setenv("FORGER_STEP_REPORT", "1")
    set_report_model(None)
    _make_step()([{"solution": CODE}])


def test_unbuildable_env_model_falls_back_quietly(monkeypatch, capsys):
    """A model that cannot be built leaves the structural summary in place."""

    def broken(*args, **kwargs):
        raise ValueError("unknown model")

    monkeypatch.setattr(model_module, "Model", broken)
    monkeypatch.setenv("FORGER_REPORT_MODEL", "nope/none")
    monkeypatch.setenv("FORGER_STEP_REPORT", "1")
    _make_step()([{"solution": CODE}])
    assert "python)" in capsys.readouterr().out
