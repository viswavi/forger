"""Completions cut off at max_tokens: counted, reported, cached, and not retried."""

from __future__ import annotations

import logging
import pickle
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pyforger import _tracking, op, pipeline, step
from pyforger.acquire.model import GenerateResult, Model
from pyforger.alter import generate_from_context
from pyforger.filter import by_predicate


class Completions:
    """Fake litellm.completion: `script[q]` lists (text, finish_reason) answers.

    `q` is the prompt's first line (output_schema adds JSON instructions after
    it). Prompts without a script answer '{"x": 1}' with finish_reason "stop".
    """

    def __init__(self):
        """Start with no scripted answers."""
        self.script: dict[str, list[tuple[str | None, str]]] = {}
        self.calls: list[str] = []
        self.lock = threading.Lock()

    def __call__(self, messages, n=1, **kwargs):
        """Fake litellm.completion."""
        text = messages[-1]["content"].split("\n")[0]
        with self.lock:
            self.calls.append(text)
            answers = [
                self.script[text].pop(0)
                if self.script.get(text)
                else ('{"x": 1}', "stop")
                for _ in range(n)
            ]
        choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None),
                finish_reason=reason,
            )
            for content, reason in answers
        ]
        return SimpleNamespace(
            choices=choices,
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=20 * n,
                completion_tokens_details=None,
            ),
            _hidden_params={"response_cost": 0.001 * n},
        )


@pytest.fixture
def fake():
    """Patch litellm in pyforger.acquire.model with a Completions fake."""
    completions = Completions()
    with patch("pyforger.acquire.model.litellm") as mock:
        mock.get_supported_openai_params.return_value = None
        mock.completion.side_effect = completions
        yield completions


def gpt() -> Model:
    """A Model with a small max_tokens."""
    return Model("gpt-4o", show_progress=False, max_tokens=100)


@op(name="alter.ask", cache=False)
def _ask(prompts, model):
    """Send `prompts` from inside an op, one row per prompt."""
    return model.generate_batch(prompts, cache_rows=prompts)


def slot_files(tmp_path: Path) -> list[Path]:
    """Every request-cache slot file under the test's FORGER_CACHE."""
    return sorted((tmp_path / "requests").glob("*/*.json"))


class TestFinishReasons:
    """GenerateResult carries each completion's finish reason."""

    def test_finish_reasons_and_truncated_count(self, fake):
        """A "length" stop is counted in `truncated`."""
        fake.script = {"Q": [("partial", "length")], "R": [("done", "stop")]}
        q, r = gpt().generate_batch(["Q", "R"])
        assert (q.finish_reasons, q.truncated) == (["length"], 1)
        assert (r.finish_reasons, r.truncated) == (["stop"], 0)

    def test_pickle_keeps_finish_reasons(self):
        """A pickled result keeps them; one built without them has None per slot."""
        result = GenerateResult(
            ["a", "b"],
            usage={},
            reasoning=[None, None],
            model_id="m",
            raw=None,
            finish_reasons=["stop", "length"],
        )
        assert pickle.loads(pickle.dumps(result)).finish_reasons == ["stop", "length"]
        bare = GenerateResult(["a"], usage={}, reasoning=[None], model_id="m", raw=None)
        assert bare.finish_reasons == [None] and bare.truncated == 0

    def test_cached_completion_keeps_its_finish_reason(self, fake):
        """A replayed truncated completion still counts as truncated."""
        fake.script = {"Q": [("partial", "length")]}
        model = gpt()
        _ask(["Q"], model)
        replay = _ask(["Q"], model)[0]
        assert fake.calls == ["Q"]
        assert replay.finish_reasons == ["length"]


class TestCaching:
    """Empty text that ran out of tokens is cached; other empty text is not."""

    @pytest.mark.parametrize("content", ["", None])
    def test_empty_length_stop_is_cached(self, fake, tmp_path, content):
        """At the same max_tokens a rerun would stop there again, so replay it."""
        fake.script = {"Q": [(content, "length")]}
        model = gpt()
        assert _ask(["Q"], model)[0] == [""]
        assert len(slot_files(tmp_path)) == 1
        assert _ask(["Q"], model)[0] == [""]
        assert fake.calls == ["Q"]

    def test_empty_stop_is_asked_again(self, fake, tmp_path):
        """Empty text with a normal stop is a glitch: not cached."""
        fake.script = {"Q": [("", "stop")]}
        _ask(["Q"], gpt())
        assert slot_files(tmp_path) == []


class TestReport:
    """The op that made the calls gets one warning per (model, max_tokens)."""

    def test_step_report_counts_truncated_completions(self, fake, monkeypatch):
        """Retries add to the same count; the warning names the cap."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        fake.script = {"Q": [("", "length")], "R": [("{", "length")]}

        @step
        def gen(rows: list[dict]) -> list[dict]:
            out = generate_from_context(
                rows, "{q}", gpt(), output_schema={"x": "int"}, merge=True
            )
            return by_predicate(out, lambda r: True)

        @pipeline
        def run():
            return gen([{"q": "Q"}, {"q": "R"}, {"q": "S"}])

        assert run() == [{"q": "S", "x": 1}]

        warnings = _tracking.step_records()["gen"].ops[0].warnings
        # The warning leads with what happened to the rows.
        assert warnings == [
            "alter.generate_from_context: 2 rows dropped (no parseable output), "
            "because 2 of 3 completions from gpt-4o stopped at max_tokens=100 "
            "(finish_reason=length) (a reasoning model's is often empty: the budget "
            "went to thinking); raise the Model's max_tokens or lower its "
            "reasoning_effort"
        ]

    def test_logged_outside_ops(self, fake, caplog):
        """A direct generate_batch call has no report, so it logs the count."""
        fake.script = {"Q": [("partial", "length")]}
        with caplog.at_level(logging.WARNING, logger="pyforger.acquire.model"):
            gpt().generate_batch(["Q", "R"])
        assert "1 of 2 completions from gpt-4o stopped at max_tokens=100" in caplog.text

    def test_no_warning_without_truncation(self, fake, monkeypatch):
        """Normal stops add nothing."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")

        @step
        def gen(rows: list[dict]) -> list[dict]:
            return generate_from_context(rows, "{q}", gpt(), merge=True)

        gen([{"q": "Q"}])

        assert _tracking.step_records()["gen"].ops[0].warnings == []


class TestRetries:
    """generate_from_context does not re-ask samples cut off at max_tokens."""

    def test_truncated_parse_failure_is_not_retried(self, fake, caplog):
        """The slot is None after one request; the log says why."""
        fake.script = {"Q": [('{"x": ', "length")], "R": [("not json", "stop")] * 4}
        with caplog.at_level(logging.WARNING):
            out = generate_from_context(
                [{"q": "Q"}, {"q": "R"}], "{q}", gpt(), output_schema={"x": "int"}
            )
        assert out == [None, None]
        assert fake.calls.count("Q") == 1
        assert fake.calls.count("R") == 4  # 1 + max_retries
        assert "1 sample(s) stopped at max_tokens" in caplog.text
        assert "1 sample(s) failed to parse after 3 retries" in caplog.text

    def test_retry_truncated_retries_them(self, fake):
        """retry_truncated=True re-asks, and a later full answer is kept."""
        fake.script = {"Q": [('{"x": ', "length"), ('{"x": 2}', "stop")]}
        out = generate_from_context(
            [{"q": "Q"}],
            "{q}",
            gpt(),
            output_schema={"x": "int"},
            retry_truncated=True,
        )
        assert out == [{"x": 2}]
        assert fake.calls.count("Q") == 2


class TestRowOutcomes:
    """generate_from_context's warnings say what happened to the rows."""

    def test_cut_off_rows_kept_without_schema(self, fake, monkeypatch):
        """Without output_schema any text is kept, so the warning says it is cut off."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        fake.script = {"Q": [("partial code", "length")]}

        @step
        def gen(rows: list[dict]) -> list[dict]:
            return generate_from_context(rows, "{q}", gpt(), merge=True)

        assert gen([{"q": "Q"}, {"q": "R"}])[0]["text"] == "partial code"
        warning = _tracking.step_records()["gen"].ops[0].warnings[0]
        assert warning.startswith(
            "alter.generate_from_context: 1 row kept with cut-off text, because 1 "
            "of 2 completions from gpt-4o stopped at max_tokens=100"
        )

    def test_dropped_and_kept_in_one_warning(self, fake, monkeypatch):
        """Unparseable cut-offs are dropped rows; parseable ones are kept, cut off."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        fake.script = {"Q": [('{"x": ', "length")], "R": [('{"x": 3} and', "length")]}

        @step
        def gen(rows: list[dict]) -> list[dict]:
            return generate_from_context(
                rows, "{q}", gpt(), output_schema={"x": "int"}, merge=True
            )

        assert gen([{"q": "Q"}, {"q": "R"}]) == [{"q": "R", "x": 3}]
        warning = _tracking.step_records()["gen"].ops[0].warnings[0]
        assert warning.startswith(
            "alter.generate_from_context: 1 row dropped (no parseable output), 1 row "
            "kept with cut-off text, because 2 of 2 completions"
        )

    def test_outcome_leads_and_mid_thought_is_counted(self, fake, monkeypatch):
        """With an outcome the warning leads with it and counts mid-thought cuts."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        fake.script = {"Q": [("<think>still thinking", "length")]}

        @step
        def gen(rows: list[dict]) -> list[dict]:
            return generate_from_context(
                rows, "{q}", gpt(), output_schema={"x": "int"}, merge=True
            )

        assert gen([{"q": "Q"}]) == []
        warning = _tracking.step_records()["gen"].ops[0].warnings[0]
        assert warning.startswith(
            "alter.generate_from_context: 1 row dropped (no parseable output), "
            "because 1 of 1 completions from gpt-4o stopped at max_tokens=100 "
            "(finish_reason=length; 1 of them mid-thought, <think> never closed) "
            "(a reasoning model's is often empty"
        )

    def test_message_takes_outcome_and_mid_thought(self):
        """truncation_message puts the outcome first, then the counts, then advice."""
        message = _tracking.truncation_message(
            "qwen",
            16384,
            78,
            264,
            "; raise the TinkerModel's max_tokens",
            outcome="20 rows dropped",
            mid_thought=72,
        )
        assert message == (
            "20 rows dropped, because 78 of 264 completions from qwen stopped at "
            "max_tokens=16384 (finish_reason=length; 72 of them mid-thought, <think> "
            "never closed); raise the TinkerModel's max_tokens"
        )
        plain = _tracking.truncation_message("qwen", 16384, 78, 264, "; raise it")
        assert plain.endswith(
            "(finish_reason=length), so their text is cut off; raise it"
        )

    def test_parse_failures_reach_the_report(self, fake, monkeypatch):
        """A sample that never parses is a data problem in the report, not a log."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        fake.script = {"R": [("not json", "stop")] * 4}

        @step
        def gen(rows: list[dict]) -> list[dict]:
            return generate_from_context(
                rows, "{q}", gpt(), output_schema={"x": "int"}, merge=True
            )

        gen([{"q": "Q"}, {"q": "R"}])
        assert _tracking.step_records()["gen"].ops[0].warnings == [
            "alter.generate_from_context: 1 sample(s) failed to parse after 3 "
            "retries → 1 row dropped"
        ]


class TestFormatTalk:
    """Reasoning that opens by negotiating the reply format is a data problem."""

    TALK = (
        '<think>We need answer JSON only? User says code block but...</think>{"x": 1}'
    )
    PLAIN = '<think>Sort the array, then scan once.</think>{"x": 1}'

    def _run(self, fake, monkeypatch, texts: list[str]) -> list[str]:
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        fake.script = {f"Q{i}": [(t, "stop")] for i, t in enumerate(texts)}

        @step
        def gen(rows: list[dict]) -> list[dict]:
            return generate_from_context(
                rows, "{q}", gpt(), output_schema={"x": "int"}, merge=True
            )

        gen([{"q": f"Q{i}"} for i in range(len(texts))])
        return _tracking.step_records()["gen"].ops[0].warnings

    def test_warns_when_many_traces_talk_about_the_format(self, fake, monkeypatch):
        """3 of 6 traces discuss JSON: the warning quotes one and says what to do."""
        warnings = self._run(fake, monkeypatch, [self.TALK] * 3 + [self.PLAIN] * 3)
        assert len(warnings) == 1
        assert warnings[0].startswith(
            "alter.generate_from_context: 3 of 6 reasoning traces open by discussing "
            'the output format (e.g. "We need answer JSON only?'
        )
        assert "generate without output_schema" in warnings[0]

    def test_quiet_below_the_share_or_count(self, fake, monkeypatch):
        """1 of 6 traces, or fewer than 5 traces in all, is not worth a warning."""
        assert self._run(fake, monkeypatch, [self.TALK] + [self.PLAIN] * 5) == []
        assert self._run(fake, monkeypatch, [self.TALK] * 4) == []

    def test_no_check_without_output_schema(self, fake, monkeypatch):
        """Without output_schema Forger added no format instruction to conflict."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        fake.script = {f"Q{i}": [(self.TALK, "stop")] for i in range(6)}

        @step
        def gen(rows: list[dict]) -> list[dict]:
            return generate_from_context(rows, "{q}", gpt(), merge=True)

        gen([{"q": f"Q{i}"} for i in range(6)])
        assert _tracking.step_records()["gen"].ops[0].warnings == []
