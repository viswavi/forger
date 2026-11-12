"""Tests for the per-completion request cache behind Model.generate_batch."""

from __future__ import annotations

import json
import logging
import math
import multiprocessing
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import litellm
import pytest

from pyforger import _tracking, filter, op, step
from pyforger.acquire import _request_cache, sample_from_LLM
from pyforger.acquire.model import Model, ReplayedRowError, _cache_kwargs, _is_row_error
from pyforger.alter import generate_from_context
from pyforger.inspect import cluster
from pyforger.inspect.cluster import _assign_topics, _chat


class GatewayTimeout(RuntimeError):
    """A gateway 504: transient, like litellm's errors with a 5xx status_code."""

    status_code = 504


class Killed(BaseException):
    """Stands in for a kill: not an Exception, so nothing catches it."""


class Completions:
    """Fake litellm.completion whose every completion text is unique.

    The k-th completion overall for content "Q" is "Q#k", so a test can tell a
    replayed completion from a new one. Contents in `fail` raise a 504 the
    first time they are asked; contents in `kill` raise Killed after `delay`;
    contents in `block` raise a content-filter 400 every time they are asked,
    and contents in `filtered` get "" with finish_reason "content_filter".
    `script[content]` lists texts to answer with instead, one per completion;
    `tool_calls=True` adds a tool call to every choice.
    """

    def __init__(self, fail=(), kill=(), delay=0.0):
        """Answer every request unless its content is in `fail` or `kill`."""
        self.fail = set(fail)
        self.kill = set(kill)
        self.block: set[str] = set()
        self.filtered: set[str] = set()  # replies the content filter cuts off
        self.delay = delay
        self.script: dict[str, list[str]] = {}
        self.tool_calls = False
        self.max_choices: int | None = None  # a gateway that ignores n
        self.lock = threading.Lock()
        self.counter = 0
        self.calls: list[tuple[str, int]] = []  # (content, n) per request

    def _text(self, prompt: str, k: int) -> str:
        if prompt in self.script:
            return self.script[prompt].pop(0)
        return f"{prompt}#{k}"

    def __call__(self, messages, n=1, **kwargs):
        """Fake litellm.completion."""
        text = messages[-1]["content"]
        with self.lock:
            self.calls.append((text, n))
            first = self.counter
            self.counter += n
            failing = text in self.fail
            self.fail.discard(text)
            n = n if self.max_choices is None else min(n, self.max_choices)
            texts = [self._text(text, first + j) for j in range(n)]
        if text in self.kill:
            time.sleep(self.delay)
            raise Killed(text)
        if failing:
            raise GatewayTimeout(f"504 for {text}")
        if text in self.block:
            raise litellm.BadRequestError(
                message=f"{{'code': 'content_filter'}} blocked {text}",
                llm_provider="azure",
                model="m",
            )
        choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content="" if text in self.filtered else texts[j],
                    reasoning_content=f"thinking {first + j}",
                    tool_calls=[{"id": "call_1"}] if self.tool_calls else None,
                ),
                finish_reason="content_filter" if text in self.filtered else "stop",
            )
            for j in range(n)
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

    def sent(self) -> list[tuple[str, int]]:
        """Requests sent so far, sorted (threads finish in any order)."""
        return sorted(self.calls)

    def reset(self) -> None:
        """Forget the requests sent so far (the counter keeps going)."""
        self.calls.clear()


@pytest.fixture
def fake():
    """Patch litellm in pyforger.acquire.model with a Completions fake."""
    completions = Completions()
    with patch("pyforger.acquire.model.litellm") as mock:
        mock.get_supported_openai_params.return_value = None
        mock.completion.side_effect = completions
        yield completions


def gpt(**kwargs) -> Model:
    """A Model whose requests may carry n > 1."""
    return Model("gpt-4o", show_progress=False, **kwargs)


@op(name="alter.ask", category="alter", cache=False, samples_arg="n")
def _ask(prompts, model, n=1, **kwargs):
    """Send `prompts` from inside an op, one row per prompt."""
    return model.generate_batch(prompts, n=n, cache_rows=prompts, **kwargs)


def ask(model, prompts, **kwargs):
    """model.generate_batch inside an op (the request cache is off outside ops).

    Uncached at the op level, so only the request cache can skip a request.
    """
    return _ask(prompts, model, **kwargs)


def ask1(model, prompt, **kwargs):
    """model.generate inside an op."""
    return ask(model, [prompt], **kwargs)[0]


def slot_files(tmp_path: Path) -> list[Path]:
    """Every request-cache slot file under the test's FORGER_CACHE."""
    return sorted((tmp_path / "requests").glob("*/*.json"))


class TestHitsAndMisses:
    """Each completion slot is cached and reused on its own."""

    def test_off_outside_ops(self, fake, tmp_path):
        """Two direct model.generate calls with one prompt both send requests."""
        model = gpt()
        first = model.generate("Q")
        second = model.generate("Q")
        assert fake.calls == [("Q", 1), ("Q", 1)]
        assert first != second
        assert slot_files(tmp_path) == []

    def test_off_in_threads_without_the_context(self, fake, tmp_path):
        """A worker thread that did not copy the context has no op scope."""
        model = gpt()

        @op(name="alter.threaded", category="alter", cache=False)
        def threaded(prompts, model):
            with ThreadPoolExecutor(1) as pool:
                return pool.submit(model.generate_batch, prompts).result()

        threaded(["Q"], model)
        threaded(["Q"], model)
        assert len(fake.calls) == 2
        assert slot_files(tmp_path) == []

    def test_second_call_sends_nothing(self, fake, tmp_path):
        """A repeated call replays every completion and costs nothing."""
        model = gpt()
        first = ask(model, ["A", "B"])
        assert fake.sent() == [("A", 1), ("B", 1)]
        fake.reset()
        second = ask(model, ["A", "B"])
        assert fake.calls == []
        assert second == first
        assert second[0].reasoning == first[0].reasoning
        assert math.isnan(second[0].usage.pop("token_units"))  # gpt-4o: no weights
        assert second[0].usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "cost_usd": 0.0,
            "cached_completions": 1,
        }
        assert second[0].raw is None
        assert len(model.history) == 4
        assert model.total_cost == pytest.approx(0.002)
        assert len(slot_files(tmp_path)) == 2

    def test_refilled_short_response_is_cached_per_slot(self, fake):
        """A gateway that ignores n: 1 choice for n=3, 2 refills, all cached."""
        fake.max_choices = 1
        model = gpt()
        three = ask1(model, "Q", n=3)
        assert fake.calls == [("Q", 3), ("Q", 1), ("Q", 1)]
        assert len(set(three)) == 3
        fake.reset()
        assert ask1(model, "Q", n=3) == three
        assert fake.calls == []

    def test_hits_add_no_step_usage(self, fake):
        """Only freshly sent responses reach the step report's usage."""
        model = gpt()
        ask(model, ["A", "B"])
        before = _tracking.usage_snapshot()
        ask(model, ["A", "B", "C"])
        after = _tracking.usage_snapshot()
        assert after[0] - before[0] == 10  # one new request's prompt tokens
        assert after[2] - before[2] == pytest.approx(0.001)

    def test_partial_reuse(self, fake):
        """3 of 5 prompts cached: a call for all 5 sends only the other 2."""
        model = gpt()
        cached = ask(model, ["P0", "P1", "P2"])
        fake.reset()
        results = ask(model, ["P0", "P1", "P2", "P3", "P4"])
        assert fake.sent() == [("P3", 1), ("P4", 1)]
        assert results[:3] == cached
        assert results[3][0].startswith("P3#")

    def test_logs_reused_count_per_batch_at_debug(self, fake, caplog):
        """Each batch logs its reuse at DEBUG only (the op logs the INFO line)."""
        model = gpt()
        ask(model, ["P0", "P1", "P2"])
        with caplog.at_level(logging.DEBUG, logger="pyforger.acquire.model"):
            ask(model, ["P0", "P1", "P2", "P3", "P4"])
        records = [
            r
            for r in caplog.records
            if r.name == "pyforger.acquire.model" and "request cache" in r.message
        ]
        assert [(r.levelno, r.message) for r in records] == [
            (logging.DEBUG, "request cache: 3 of 5 completions reused (gpt-4o)")
        ]

    def test_n_is_topped_up(self, fake):
        """n=4 cached, then n=6: one request for the 2 missing slots."""
        model = gpt()
        four = ask1(model, "Q", n=4)
        fake.reset()
        six = ask1(model, "Q", n=6)
        assert fake.calls == [("Q", 2)]
        assert six[:4] == four
        assert len(set(six)) == 6
        assert six.usage["cached_completions"] == 4
        assert six.usage["completion_tokens"] == 40  # only the 2 new ones

    def test_fan_out_is_topped_up(self, fake):
        """On a model without n support the missing slots go as n=1 requests."""
        model = Model("xai/grok-4", show_progress=False)
        ask1(model, "Q", n=4)
        fake.reset()
        six = ask1(model, "Q", n=6)
        assert fake.calls == [("Q", 1), ("Q", 1)]
        assert len(set(six)) == 6

    def test_smaller_n_reads_a_prefix(self, fake):
        """n=2 after n=4 reuses slots 0 and 1."""
        model = gpt()
        four = ask1(model, "Q", n=4)
        fake.reset()
        assert ask1(model, "Q", n=2) == four[:2]
        assert fake.calls == []

    def test_large_n_split_and_cached(self, fake):
        """n=20 is split into 10 + 10; a rerun sends nothing; n=40 sends 20."""
        model = gpt()
        twenty = ask1(model, "Q", n=20)
        assert fake.calls == [("Q", 10), ("Q", 10)]
        fake.reset()
        assert ask1(model, "Q", n=20) == twenty
        assert fake.calls == []
        forty = ask1(model, "Q", n=40)
        assert fake.sent() == [("Q", 10), ("Q", 10)]
        assert forty[:20] == twenty
        assert len(set(forty)) == 40


class TestSlots:
    """Repeated prompts in one call and retries get distinct completions."""

    def test_repeated_prompt_gets_distinct_slots(self, fake):
        """Three copies of one prompt (three judges) get three completions."""
        model = gpt()
        first = ask(model, ["J", "J", "J"])
        assert len({r[0] for r in first}) == 3
        fake.reset()
        assert ask(model, ["J", "J", "J"]) == first
        assert fake.calls == []
        # The copies used slots 0, 1, 2: the same completions as n=3.
        assert ask1(model, "J", n=3) == [r[0] for r in first]
        assert fake.calls == []

    def test_repeated_prompt_with_n(self, fake):
        """With n=2 the second copy uses slots 2 and 3."""
        model = gpt()
        pair = ask(model, ["J", "J"], n=2)
        assert len(set(pair[0] + pair[1])) == 4
        fake.reset()
        assert ask1(model, "J", n=4) == pair[0] + pair[1]
        assert fake.calls == []

    def test_salt_gets_fresh_completions(self, fake):
        """A new cache_salt asks again; the same salt replays."""
        model = gpt()
        plain = ask1(model, "Q")
        salted = ask1(model, "Q", cache_salt="retry 1")
        assert salted != plain
        fake.reset()
        assert ask1(model, "Q", cache_salt="retry 1") == salted
        assert fake.calls == []

    def test_generate_from_context_retry_is_fresh(self, fake):
        """A parse-failure retry does not get the failed completion back."""
        model = gpt()
        outputs = iter(["not json", '{"answer": "4"}'])
        fake_call = fake.__call__

        def scripted(messages, n=1, **kwargs):
            response = fake_call(messages, n=n, **kwargs)
            for choice in response.choices:
                choice.message.content = next(outputs)
            return response

        with patch("pyforger.acquire.model.litellm.completion", side_effect=scripted):
            rows = generate_from_context(
                [{"q": "2+2"}],
                "Answer {q} as JSON.",
                model=model,
                output_schema={"answer": "str"},
                max_retries=1,
            )
        assert len(fake.calls) == 2
        assert rows[0]["answer"] == "4"

    def test_sample_from_llm_rounds_are_fresh(self, fake):
        """Each sample_from_LLM round asks for new completions."""
        model = gpt()
        rows = sample_from_LLM("Write a line.", model, n=4, batch_size=2)
        assert len(fake.calls) == 2
        assert len({row["text"] for row in rows}) == 4

    def test_sample_from_llm_pools_overlap_only_if_only_n_differs(self, fake):
        """Pools of n=4 and n=2 overlap unless another argument differs.

        n is left out of the scope so that raising it tops up (n=6 sends 2).
        """
        model = gpt()
        prompt = "Write a question."
        train = sample_from_LLM(prompt, model, n=4)
        test = sample_from_LLM(prompt, model, n=2)
        assert test == train[:2]
        fake.reset()
        assert sample_from_LLM(prompt, model, n=6)[:4] == train
        assert fake.calls == [(prompt, 2)]
        fake.reset()
        named = sample_from_LLM(prompt, model, n=2, name="test pool")
        hotter = sample_from_LLM(prompt, model, n=2, temperature=0.9)
        assert len(fake.calls) == 2
        texts = {r["text"] for r in train}
        assert texts.isdisjoint(r["text"] for r in named + hotter)


class TestBypass:
    """_no_cache, Model(request_cache=False), and FORGER_REQUEST_CACHE=0."""

    def test_no_cache_op_skips_reads_but_writes(self, fake, tmp_path):
        """_no_cache=True asks again and stores what it got."""
        model = gpt()

        @op
        def ask(prompts, model):
            return [list(r) for r in model.generate_batch(prompts)]

        first = ask(["Q"], model)
        fresh = ask(["Q"], model, _no_cache=True)
        assert len(fake.calls) == 2
        assert fresh != first
        [path] = slot_files(tmp_path)
        assert json.loads(path.read_text())["text"] == fresh[0][0]

    def test_model_flag_off(self, fake, tmp_path):
        """request_cache=False neither reads nor writes."""
        model = gpt(request_cache=False)
        ask1(model, "Q")
        ask1(model, "Q")
        assert len(fake.calls) == 2
        assert slot_files(tmp_path) == []

    def test_env_off(self, fake, tmp_path, monkeypatch):
        """FORGER_REQUEST_CACHE=0 turns it off for every Model."""
        monkeypatch.setenv("FORGER_REQUEST_CACHE", "0")
        model = gpt()
        ask1(model, "Q")
        ask1(model, "Q")
        assert len(fake.calls) == 2
        assert slot_files(tmp_path) == []

    def test_flag_not_in_cache_key(self):
        """Turning the request cache off keeps @op cache hits."""
        assert gpt().cache_key() == gpt(request_cache=False).cache_key()


class TestFailures:
    """Errors are never cached; finished requests are kept.

    Models here make one attempt (num_retries=0): the fake's 504 is what a
    request gives once Model's retries run out.
    """

    def test_errors_not_cached(self, fake, tmp_path):
        """A 504 leaves its slot empty; the rerun sends only that request."""
        fake.fail = {"B"}
        model = gpt(num_retries=0)
        results = ask(model, ["A", "B", "C"], return_exceptions=True)
        assert isinstance(results[1], RuntimeError)
        assert len(slot_files(tmp_path)) == 2
        fake.reset()
        again = ask(model, ["A", "B", "C"], return_exceptions=True)  # same op args
        assert fake.calls == [("B", 1)]
        assert again[0] == results[0] and again[2] == results[2]

    def test_single_request_error_not_cached(self, fake, tmp_path):
        """A failed one-request call writes nothing."""
        fake.fail = {"A"}
        with pytest.raises(RuntimeError):
            ask1(gpt(num_retries=0), "A")
        assert slot_files(tmp_path) == []

    def test_raised_failure_keeps_finished(self, fake):
        """Without return_exceptions the batch raises, but finished ones stay."""
        fake.fail = {"P2"}
        model = gpt(num_retries=0)
        with pytest.raises(RuntimeError):
            ask(model, [f"P{i}" for i in range(5)])
        fake.reset()
        ask(model, [f"P{i}" for i in range(5)])
        assert fake.calls == [("P2", 1)]

    def test_killed_run_keeps_finished(self, fake):
        """A kill mid-batch: requests that finished before it are on disk."""
        fake.kill = {"SLOW"}
        fake.delay = 0.2
        model = gpt()
        prompts = ["SLOW", "P1", "P2", "P3"]
        with pytest.raises(Killed):
            ask(model, prompts)
        assert len(model.history) == 0  # the batch never returned
        fake.kill = set()
        fake.reset()
        ask(model, prompts)
        assert fake.calls == [("SLOW", 1)]

    def test_used_up_quota_replays_cached_prompts(self, fake, monkeypatch, caplog):
        """With the quota used up, cached prompts replay and the others drop out.

        Nothing is sent, the warning names the unsent prompts, and the op is not
        cached; an op with nothing to replay raises the quota error.
        """
        from pyforger.op import trace
        from pyforger.token_units import TokenUnitQuotaExceeded

        rows = [{"q": "A"}, {"q": "B"}, {"q": "C"}]
        first = generate_from_context(rows[:2], "{q}", gpt(num_retries=0))

        def used_up(self):
            raise TokenUnitQuotaExceeded("Token-unit quota exhausted")

        # The check Model makes before sending each request (not on a cache hit).
        monkeypatch.setattr(Model, "_before_request", used_up)
        fake.reset()
        with caplog.at_level(logging.WARNING):
            again = generate_from_context(rows, "{q}", gpt(num_retries=0))
        assert fake.calls == []
        assert again == [*first, None]
        assert "1 prompt not sent: the token-unit quota is used up" in caplog.text
        assert trace()[-1].cache_hit is False
        with pytest.raises(TokenUnitQuotaExceeded):
            generate_from_context(rows[2:], "{q}", gpt(num_retries=0))
        assert fake.calls == []

    def test_partial_fan_out_with_hits(self, fake):
        """Hits plus a failed top-up request: the hits come back, fewer than n."""
        model = Model("xai/grok-4", show_progress=False, num_retries=0)
        two = ask1(model, "Q", n=2, return_exceptions=True)
        fake.fail = {"Q"}
        [result] = ask(model, ["Q"], n=3, return_exceptions=True)
        assert list(result) == list(two)


class TestRowErrors:
    """A content-filter block is cached as one, so a rerun gives the same rows.

    The filter does not block a prompt every time: asking again could answer
    it, and that one new row would change every later step's input.
    """

    def test_blocked_prompt_is_replayed_not_asked_again(self, fake, tmp_path):
        """The rerun gets the block back without a request, though B would pass."""
        fake.block = {"B"}
        model = gpt(num_retries=0)
        first = ask(model, ["A", "B", "C"], return_exceptions=True)
        assert isinstance(first[1], litellm.BadRequestError)
        assert len(slot_files(tmp_path)) == 3  # A, C, and B's block
        fake.block = set()
        fake.reset()
        again = ask(model, ["A", "B", "C"], return_exceptions=True)
        assert fake.calls == []
        assert again[0] == first[0] and again[2] == first[2]
        assert isinstance(again[1], ReplayedRowError)
        assert again[1].reason == "content filter" and _is_row_error(again[1])
        assert "_no_cache=True" in str(again[1])

    def test_no_cache_asks_again(self, fake):
        """_no_cache=True sends the blocked prompt again and keeps its answer."""
        fake.block = {"B"}
        model = gpt(num_retries=0)
        ask(model, ["B"], return_exceptions=True)
        fake.block = set()
        fake.reset()
        [answer] = _ask(["B"], model, return_exceptions=True, _no_cache=True)
        assert fake.calls == [("B", 1)]
        assert isinstance(answer, list) and answer[0].startswith("B#")
        fake.reset()
        [replay] = ask(model, ["B"], return_exceptions=True)
        assert list(replay) == list(answer)  # the answer replaced the block
        assert fake.calls == []

    def test_replay_raises_without_return_exceptions(self, fake):
        """Without return_exceptions, the replayed block raises as the first did."""
        fake.block = {"B"}
        model = gpt(num_retries=0)
        with pytest.raises(litellm.BadRequestError):
            ask(model, ["A", "B"])
        fake.block = set()
        fake.reset()
        with pytest.raises(ReplayedRowError, match="content_filter"):
            ask(model, ["A", "B"])
        assert fake.calls == []

    def test_blocked_slot_of_a_fanned_out_prompt_replays_the_block(self, fake):
        """One blocked copy of n fails the whole prompt, on the rerun too."""
        model = Model("xai/grok-4", show_progress=False, num_retries=0)
        ask1(model, "Q", n=1)  # slot 0 answered
        fake.block = {"Q"}
        [first] = ask(model, ["Q"], n=2, return_exceptions=True)  # slot 1 blocked
        assert isinstance(first, litellm.BadRequestError)
        fake.block = set()
        fake.reset()
        [again] = ask(model, ["Q"], n=2, return_exceptions=True)
        assert isinstance(again, ReplayedRowError)
        assert fake.calls == []

    def test_reply_cut_by_the_filter_is_replayed(self, fake):
        """An empty reply the filter stopped replays as is, not asked again."""
        fake.filtered = {"B"}
        model = gpt(num_retries=0)
        first = ask(model, ["B"])[0]
        assert list(first) == [""] and first.finish_reasons == ["content_filter"]
        fake.filtered = set()
        fake.reset()
        again = ask(model, ["B"])[0]
        assert fake.calls == []
        assert list(again) == [""] and again.finish_reasons == ["content_filter"]

    def test_op_with_only_blocked_rows_is_cached(self, fake, caplog):
        """generate_from_context caches its result; a 504 still keeps it uncached."""
        from pyforger.op import trace

        rows = [{"q": "A"}, {"q": "B"}, {"q": "C"}]
        fake.block = {"B"}
        with caplog.at_level(logging.WARNING):
            first = generate_from_context(rows, "{q}", gpt(num_retries=0))
        assert first[1] is None
        assert "(result cached, so a rerun gives the same rows" in caplog.text
        fake.block = set()
        fake.reset()
        assert generate_from_context(rows, "{q}", gpt(num_retries=0)) == first
        assert trace()[-1].cache_hit is True
        assert fake.calls == []

    def test_block_and_504_rerun_sends_only_the_504(self, fake):
        """With a 504 too the op is not cached; its rerun re-sends only that row."""
        from pyforger.op import trace

        rows = [{"q": "A"}, {"q": "B"}, {"q": "C"}]
        fake.block, fake.fail = {"A"}, {"B"}
        first = generate_from_context(rows, "{q}", gpt(num_retries=0))
        assert first[0] is None and first[1] is None
        fake.block = set()
        fake.reset()
        again = generate_from_context(rows, "{q}", gpt(num_retries=0))
        assert trace()[-1].cache_hit is False
        assert fake.calls == [("B", 1)]
        assert again[0] is None and again[1]["text"].startswith("B#")
        assert again[2] == first[2]


class TestKey:
    """What goes into the key."""

    def test_key_ignores_credentials_and_transport(self, fake):
        """api_key, timeouts, retries, and metadata do not change the key."""
        first = ask1(gpt(api_key="sk-a", timeout=5), "Q")
        again = ask1(
            gpt(api_key="sk-b", timeout=60, num_retries=0, metadata={"run": 2}), "Q"
        )
        assert again == first
        assert len(fake.calls) == 1

    @pytest.mark.parametrize(
        "server",
        [
            {"api_base": "http://ckpt-b"},
            {"base_url": "http://ckpt-b"},
            {"api_version": "2025-01-01"},
            {"custom_llm_provider": "hosted_vllm"},
        ],
    )
    def test_key_changes_with_server(self, fake, server):
        """Another server can serve other weights under the same model name."""
        ask1(Model("hosted_vllm/policy", api_base="http://ckpt-a"), "Q")
        ask1(
            Model("hosted_vllm/policy", **({"api_base": "http://ckpt-a"} | server)), "Q"
        )
        assert len(fake.calls) == 2

    def test_model_retries_and_timeout_keep_op_requests(self, fake):
        """Inside an op, a Model with other retries or timeout reuses requests."""
        rows = [{"q": "a"}, {"q": "b"}]
        first = generate_from_context(rows, "{q}", gpt(timeout=5))
        fake.reset()
        again = generate_from_context(rows, "{q}", gpt(timeout=60, num_retries=0))
        assert again == first
        assert fake.calls == []

    def test_cache_rows_must_align(self, fake):
        """cache_rows needs one entry per prompt."""
        with pytest.raises(ValueError, match="cache_rows"):
            gpt().generate_batch(["A", "B"], cache_rows=[{"q": 1}])

    @pytest.mark.parametrize(
        "change",
        [
            {"temperature": 0.2},
            {"max_tokens": 50},
            {"seed": 1},
            {"reasoning_effort": "low"},
            {"extra_body": {"top_k": 5}},
        ],
    )
    def test_key_changes_with_params(self, fake, change):
        """Output-changing params give a new key."""
        ask1(gpt(), "Q")
        ask1(gpt(**change), "Q")
        assert len(fake.calls) == 2

    def test_key_changes_with_messages_and_model(self, fake):
        """Different messages, system prompt, or model id: new requests."""
        ask1(gpt(), "Q")
        ask1(gpt(), "Q")
        assert len(fake.calls) == 1
        ask1(
            gpt(),
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Q"},
            ],
        )
        ask1(Model("gpt-4o-mini", show_progress=False), "Q")
        ask1(gpt(), "Q", temperature=0.1)
        assert len(fake.calls) == 4


def drop_op_cache(tmp_path: Path) -> None:
    """Delete every @op cache entry, keeping the request cache."""
    for entry in tmp_path.iterdir():
        if entry.is_dir() and entry.name != "requests":
            shutil.rmtree(entry)


def texts(rows: list[dict | None]) -> list[str]:
    """The text of each generate_from_context output row."""
    return [row["text"] for row in rows if row is not None]


def stage(label: str, template: str):
    """An op that asks `template` per row and stores the answer under `label`.

    Uncached at the op level, so only the request cache can skip a request.
    """

    @op(name=f"alter.{label}", cache=False)
    def run(rows, model):
        answers = model.generate_batch(
            [template.format(**row) for row in rows], cache_rows=rows
        )
        return [dict(row, **{label: answer[0]}) for row, answer in zip(rows, answers)]

    return run


class TestOpScope:
    """Requests are keyed on the op call and the row, not on the prompt alone."""

    def test_separate_ops_draw_separate_samples(self, fake):
        """expert1 / expert2 from one template are independent draws."""
        model = gpt()
        rows = [{"q": "6*7"}, {"q": "2+2"}]
        expert1 = generate_from_context(rows, "Solve {q}", model)
        with_expert1 = [dict(r, expert1=e["text"]) for r, e in zip(rows, expert1)]
        expert2 = generate_from_context(with_expert1, "Solve {q}", model)
        assert len(fake.calls) == 4
        assert set(texts(expert1)).isdisjoint(texts(expert2))

    def test_shared_upstream_is_reused_but_siblings_are_not(self, fake):
        """A(D(C(x))) then A(B(C(x))): C is reused, B never gets D's samples."""
        model = gpt()
        x = [{"q": "a"}, {"q": "b"}]
        a, c = stage("a", "Final {q}"), stage("c", "Start {q}")
        b, d = stage("b", "Think {q}"), stage("d", "Think {q}")  # same template
        first = a(d(c(x, model), model), model)
        assert len(fake.calls) == 6
        fake.reset()
        second = a(b(c(x, model), model), model)
        assert [r["c"] for r in second] == [r["c"] for r in first]
        assert {r["b"] for r in second}.isdisjoint(r["d"] for r in first)
        assert sorted(text for text, _ in fake.calls) == [
            "Final a",
            "Final b",
            "Think a",
            "Think b",
        ]

    def test_changing_one_row_resends_only_that_row(self, fake):
        """Editing 1 of 5 rows re-sends 1 request."""
        model = gpt()
        rows = [{"q": f"q{i}"} for i in range(5)]
        first = generate_from_context(rows, "Solve {q}", model)
        fake.reset()
        edited = [dict(r) for r in rows]
        edited[2]["q"] = "changed"
        second = generate_from_context(edited, "Solve {q}", model)
        assert fake.calls == [("Solve changed", 1)]
        assert texts(second)[:2] == texts(first)[:2]
        assert texts(second)[3:] == texts(first)[3:]

    def test_row_key_ignores_other_rows_but_not_its_own_fields(self, fake):
        """A field the prompt does not use still changes that row's key."""
        model = gpt()
        rows = [{"q": "a"}, {"q": "b"}]
        generate_from_context(rows, "Solve {q}", model)
        fake.reset()
        generate_from_context([rows[0], dict(rows[1], note="x")], "Solve {q}", model)
        assert fake.calls == [("Solve b", 1)]

    def test_editing_the_template_resends_everything(self, fake):
        """Another template is another op scope."""
        model = gpt()
        rows = [{"q": "a"}, {"q": "b"}]
        generate_from_context(rows, "Solve {q}", model)
        fake.reset()
        generate_from_context(rows, "Solve: {q}", model)
        assert len(fake.calls) == 2

    def test_editing_the_op_body_resends_everything(self, fake):
        """Two ops with other bodies but one name and template share nothing."""
        model = gpt()

        @op(name="alter.ask", category="alter", cache=False)
        def ask(rows, model):
            return model.generate_batch([r["q"] for r in rows], cache_rows=rows)

        ask([{"q": "a"}], model)

        @op(name="alter.ask", category="alter", cache=False)
        def ask(rows, model):  # noqa: F811 - an edited version of the op
            out = model.generate_batch([r["q"] for r in rows], cache_rows=rows)
            return list(out)

        ask([{"q": "a"}], model)
        assert len(fake.calls) == 2

    def test_identical_rows_get_distinct_samples(self, fake):
        """Duplicate rows in one op call differ, also across batch_size chunks."""
        model = gpt()
        rows = [{"q": "flip"}] * 3
        assert len(set(texts(generate_from_context(rows, "{q}", model)))) == 3
        chunked = generate_from_context(rows, "{q}", model, batch_size=1)
        assert len(set(texts(chunked))) == 3

    def test_rerun_replays_everything(self, fake, tmp_path):
        """With the op cache gone, a rerun sends nothing and returns the same."""
        model = gpt()
        rows = [{"q": "flip"}, {"q": "flip"}, {"q": "other"}]
        first = generate_from_context(rows, "{q}", model, batch_size=1)
        drop_op_cache(tmp_path)
        fake.reset()
        assert generate_from_context(rows, "{q}", model, batch_size=1) == first
        assert fake.calls == []

    def test_samples_per_item_tops_up(self, fake):
        """samples_per_item 2 -> 3 sends one completion per row."""
        model = gpt()
        rows = [{"q": "a"}, {"q": "b"}]
        two = generate_from_context(rows, "{q}", model, samples_per_item=2)
        fake.reset()
        three = generate_from_context(rows, "{q}", model, samples_per_item=3)
        assert fake.sent() == [("a", 1), ("b", 1)]
        assert texts(three)[0:2] == texts(two)[0:2]

    def test_by_llm_score_rows_and_judges(self, fake):
        """Judges get distinct samples; a new row costs only its own judges."""
        model = gpt()
        rows = [{"q": f"q{i}", "a": "x"} for i in range(3)]

        def judge(rows):
            return filter.by_llm_score(
                rows,
                model=model,
                input_field="q",
                output_field="a",
                num_judges=3,
                temperature=0.7,
            )

        judge(rows)
        assert len(fake.calls) == 9  # one n=1 request per row and judge
        assert len({c for c in fake.calls}) == 3  # three distinct prompts
        fake.reset()
        judge(rows + [{"q": "q3", "a": "x"}])
        assert len(fake.calls) == 3  # only the new row's three judges

    def test_logs_once_per_op_call(self, fake, caplog):
        """An op with several batches logs its reuse in one INFO line, even 0."""
        model = gpt()
        rows = [{"q": f"q{i}"} for i in range(4)]
        with caplog.at_level(logging.INFO, logger="forger.op"):
            generate_from_context(rows[:3], "{q}", model, batch_size=1)
            generate_from_context(rows, "{q}", model, batch_size=1)
        lines = [r.message for r in caplog.records if "request cache" in r.message]
        assert lines == [
            "request cache: 0 of 3 completions reused in alter.generate_from_context",
            "request cache: 3 of 4 completions reused in alter.generate_from_context",
        ]

    def test_unsupported_keywords_are_dropped(self):
        """Fakes that override generate_batch without cache keywords still work."""

        class Fake:
            def generate_batch(self, prompts, *, n=1):
                return prompts

        kwargs = _cache_kwargs(Fake().generate_batch, cache_rows=[1], cache_salt="s")
        assert kwargs == {}
        assert _cache_kwargs(gpt().generate_batch, cache_rows=[1], cache_salt=None) == {
            "cache_rows": [1]
        }


DEEPSEEK = "litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731"  # fixed unit weights


def resent_warnings(caplog) -> list[str]:
    """The "answered before" warnings logged so far."""
    return [r.message for r in caplog.records if "answered before" in r.message]


class TestResentWarning:
    """A batch whose rows were answered before under another prompt warns first."""

    @pytest.fixture(autouse=True)
    def no_quota(self, monkeypatch):
        """DEEPSEEK requests must not touch a shared token-unit ledger."""
        monkeypatch.delenv("FORGER_TOKEN_USAGE_FILE", raising=False)
        monkeypatch.delenv("FORGER_TOKEN_UNIT_LIMIT", raising=False)

    def test_edited_step_template_warns_once_with_earlier_cost(self, fake, caplog):
        """30 rows answered under "Solve", rerun as 60 under "Solve now": 30 warn.

        Each fake answer is 10 prompt + 20 completion tokens, 70 units.
        """
        model = Model(DEEPSEEK, show_progress=False)
        rows = [{"q": f"q{i}"} for i in range(60)]

        @step(name="answer")
        def answer(rows):
            return generate_from_context(rows, "Solve {q}", model)

        answer(rows[:30])

        @step(name="answer")
        def answer(rows):  # noqa: F811 - the template edited for every row
            return generate_from_context(rows, "Solve now {q}", model, batch_size=40)

        fake.reset()
        with caplog.at_level(logging.WARNING):
            answer(rows)
        # A Model gets all 60 prompts in one call whatever batch_size says.
        message = (
            "alter.generate_from_context: 30 of 60 rows were answered before under "
            "a different prompt or settings and will be asked again (the request "
            "cache keys on the exact prompt); ~2,100 token units at the earlier "
            "cost per row"
        )
        assert resent_warnings(caplog) == [message]  # once, not per chunk
        assert len(fake.calls) == 60  # a warning, never a block
        rec = _tracking.step_records()["answer"]
        assert message in rec.warnings

    def test_bare_op_warns_with_earlier_dollar_cost(self, fake, caplog):
        """Outside a step the op call is the run; no unit weights: dollars."""
        model = gpt()
        rows = [{"q": f"q{i}"} for i in range(20)]
        generate_from_context(rows, "Solve {q}", model)
        with caplog.at_level(logging.WARNING):
            generate_from_context(rows, "Solve: {q}", model)
        assert resent_warnings(caplog) == [
            "alter.generate_from_context: 20 of 20 rows were answered before under "
            "a different prompt or settings and will be asked again (the request "
            "cache keys on the exact prompt); ~$0.02 at the earlier cost per row"
        ]

    def test_new_rows_and_replays_do_not_warn(self, fake, caplog, tmp_path):
        """Rows never answered, or answered under this prompt, are not re-asked."""
        model = gpt()
        rows = [{"q": f"q{i}"} for i in range(60)]
        with caplog.at_level(logging.WARNING):
            generate_from_context(rows[:30], "Solve {q}", model)
            generate_from_context(rows, "Solve {q}", model)
            drop_op_cache(tmp_path)
            generate_from_context(rows, "Solve {q}", model)
        assert resent_warnings(caplog) == []

    def test_few_or_a_minority_of_rows_do_not_warn(self, fake, caplog):
        """19 re-asked rows, or 25 of 60, are under the thresholds."""
        model = gpt()
        few = [{"q": f"few{i}"} for i in range(19)]
        many = [{"q": f"many{i}"} for i in range(60)]
        generate_from_context(few, "Solve {q}", model)
        generate_from_context(many[:25], "Solve {q}", model)
        with caplog.at_level(logging.WARNING):
            generate_from_context(few, "Solve: {q}", model)
            generate_from_context(many, "Solve: {q}", model)
        assert resent_warnings(caplog) == []

    def test_two_ops_on_the_same_rows_in_one_step_do_not_warn(self, fake, caplog):
        """Two templates over the same rows in one step run are not an edit."""
        model = gpt()
        rows = [{"q": f"q{i}"} for i in range(30)]

        @step(name="two_views")
        def two_views(rows):
            hints = generate_from_context(rows, "Hint for {q}", model)
            answers = generate_from_context(rows, "Answer {q}", model)
            return [
                dict(r, hint=h["text"], answer=a["text"])
                for r, h, a in zip(rows, hints, answers)
            ]

        with caplog.at_level(logging.WARNING):
            two_views(rows)
        assert resent_warnings(caplog) == []
        assert len(fake.calls) == 60

    def test_no_cache_does_not_warn(self, fake, caplog):
        """_no_cache=True asks for fresh answers on purpose."""
        model = gpt()
        rows = [{"q": f"q{i}"} for i in range(20)]
        generate_from_context(rows, "Solve {q}", model)
        with caplog.at_level(logging.WARNING):
            generate_from_context(rows, "Solve: {q}", model, _no_cache=True)
        assert resent_warnings(caplog) == []


class TestNotCached:
    """What the request cache refuses to store."""

    @pytest.mark.parametrize("text", ["", None])
    def test_empty_completions_are_asked_again(self, fake, tmp_path, text):
        """An empty or missing text is a glitch, not an answer to replay."""
        fake.script = {"Q": [text, "real"]}
        model = gpt()
        assert ask1(model, "Q") == [text or ""]
        assert slot_files(tmp_path) == []
        assert ask1(model, "Q") == ["real"]

    def test_tool_call_choices_are_not_cached(self, fake, tmp_path):
        """A replay could not rebuild tool calls, so they are asked again."""
        fake.tool_calls = True
        model = gpt()
        ask1(model, "Q")
        ask1(model, "Q")
        assert len(fake.calls) == 2
        assert slot_files(tmp_path) == []

    def test_requests_with_tools_are_not_cached(self, fake, tmp_path):
        """A Model with tools= (or logprobs=) skips the request cache."""
        model = gpt(tools=[{"type": "function", "function": {"name": "f"}}])
        ask1(model, "Q")
        ask1(model, "Q")
        assert len(fake.calls) == 2
        assert slot_files(tmp_path) == []


class TestClusterRetries:
    """topic_based_clustering retries get new labels, not the invalid one."""

    def test_assign_retry_is_fresh(self, fake):
        """An invalid first label is re-asked and the valid retry is used."""
        codebook = {"Algebra": "equations", "Geometry": "shapes"}
        prompt = _assign_prompt("A triangle has three sides.", codebook)
        fake.script = {prompt: ['{"topic": "Nope"}', '{"topic": "Geometry"}']}

        @op(name="inspect.assign", category="inspect", cache=False)
        def assign(docs, model):
            return _assign_topics(
                docs, model, "math", codebook, truncation=1000, max_retries=2
            )

        topics = assign(["A triangle has three sides."], gpt())
        assert topics == {"A triangle has three sides.": "Geometry"}
        assert len(fake.calls) == 2


def _assign_prompt(doc: str, codebook: dict[str, str]) -> str:
    """The user message _assign_topics sends for `doc`."""
    return _chat(
        "",
        cluster._ASSIGN_USER.format(
            context="math", topics=json.dumps(codebook, indent=2), record=doc
        ),
    )[-1]["content"]


def _write_many(args):
    """Process worker: write the same slots many times."""
    key, worker = args
    for i in range(50):
        _request_cache.write_slot(key, i % 5, {"text": f"w{worker}-{i}"})


class TestConcurrency:
    """Atomic writes: concurrent writers never leave a broken slot."""

    def test_threads_writing_and_reading(self, tmp_path):
        """8 threads write the same 5 slots while 2 threads read them."""
        key = "ab" * 32
        stop = threading.Event()
        bad: list[object] = []

        def write(worker):
            for i in range(100):
                _request_cache.write_slot(key, i % 5, {"text": f"w{worker}-{i}"})

        def read():
            while not stop.is_set():
                for slot in range(5):
                    path = tmp_path / "requests" / key[:2] / f"{key}.{slot}.json"
                    if path.exists() and _request_cache.read_slot(key, slot) is None:
                        bad.append(slot)

        readers = [threading.Thread(target=read) for _ in range(2)]
        writers = [threading.Thread(target=write, args=(w,)) for w in range(8)]
        for t in readers + writers:
            t.start()
        for t in writers:
            t.join()
        stop.set()
        for t in readers:
            t.join()
        assert bad == []
        files = sorted(p.name for p in (tmp_path / "requests" / key[:2]).iterdir())
        assert files == [f"{key}.{slot}.json" for slot in range(5)]  # no temp files

    def test_processes_writing_the_same_slots(self, tmp_path):
        """Several processes writing one key leave valid JSON in every slot."""
        key = "cd" * 32
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(3) as pool:
            pool.map(_write_many, [(key, w) for w in range(3)])
        for slot in range(5):
            path = tmp_path / "requests" / key[:2] / f"{key}.{slot}.json"
            assert json.loads(path.read_text())["text"].startswith("w")

    def test_models_in_threads_share_the_cache(self, fake):
        """Two threads asking the same prompts both get full results."""
        model = gpt(max_concurrency=4)
        prompts = [f"Q{i}" for i in range(10)]
        results = {}
        threads = [
            threading.Thread(
                target=lambda k=k: results.setdefault(k, ask(model, prompts))
            )
            for k in ("a", "b")
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(len(r) == 10 and len(r[0]) == 1 for r in results.values())
        fake.reset()
        replay = ask(model, prompts)
        assert fake.calls == []
        assert all(len(r) == 1 for r in replay)

    def test_other_users_files_are_ignored(self, tmp_path, monkeypatch):
        """A slot file owned by another uid reads as a miss."""
        _request_cache.write_slot("ef" * 32, 0, {"text": "x"})
        assert _request_cache.read_slot("ef" * 32, 0) == {"text": "x"}
        monkeypatch.setattr("os.getuid", lambda: -1)
        assert _request_cache.read_slot("ef" * 32, 0) is None

    def test_corrupt_slot_is_a_miss(self, tmp_path):
        """An unreadable slot file is asked again, not raised."""
        path = tmp_path / "requests" / "12" / f"{'12' * 32}.0.json"
        path.parent.mkdir(parents=True)
        path.write_text("{not json")
        assert _request_cache.read_slot("12" * 32, 0) is None
