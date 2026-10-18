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
        assert warnings == [
            "alter.generate_from_context: 2 of 3 completions from gpt-4o stopped at "
            "max_tokens=100 (finish_reason=length), so their text is cut off (a "
            "reasoning model's is often empty: the budget went to thinking); raise "
            "the Model's max_tokens or lower its reasoning_effort"
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
