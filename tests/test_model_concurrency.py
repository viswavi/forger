"""Tests for Model max_concurrency, return_exceptions, progress, batch_size=None."""

from __future__ import annotations

import copy
import logging
import pickle
import sys
import threading
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fixtures import MockModel, RecordingMockModel

import pyforger.acquire.model as model_module
from pyforger import op
from pyforger.acquire import sample_from_LLM
from pyforger.acquire.model import GenerateResult, LocalModel, Model, _progress_mode
from pyforger.alter import generate_from_context, self_taught_reasoner
from pyforger.op import trace
from pyforger.utils import canonical


def make_response(text, prompt_tokens=10, completion_tokens=20, cost=0.001):
    """A litellm-shaped response with one choice."""
    choice = MagicMock()
    choice.message.content = text
    choice.message.reasoning_content = None
    response = MagicMock()
    response.choices = [choice]
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.usage.completion_tokens_details.reasoning_tokens = 0
    response._hidden_params = {"response_cost": cost}
    return response


class InFlight:
    """Fake completion that sleeps and records the most requests in flight."""

    def __init__(self, delay=0.05, fail=()):
        """Answer each request with its content after `delay` seconds.

        Requests whose content is in `fail` raise after the delay instead; a
        content of the form "P@k" fails the k-th request for "P".
        """
        self.delay = delay
        self.fail = set(fail)
        self.lock = threading.Lock()
        self.current = 0
        self.max = 0
        self.calls = 0
        self.seen: dict[str, int] = {}

    def __call__(self, messages, **kwargs):
        """Fake litellm.completion."""
        text = messages[0]["content"]
        with self.lock:
            self.current += 1
            self.calls += 1
            self.max = max(self.max, self.current)
            k = self.seen.get(text, 0)
            self.seen[text] = k + 1
        try:
            time.sleep(self.delay)
            if text in self.fail or f"{text}@{k}" in self.fail:
                raise RuntimeError(f"504 for {text}")
            return make_response(text)
        finally:
            with self.lock:
                self.current -= 1


@pytest.fixture
def mock_litellm():
    """Patch litellm in pyforger.acquire.model; registry knows no model."""
    with patch("pyforger.acquire.model.litellm") as mock:
        mock.get_supported_openai_params.return_value = None
        yield mock


def fake_terminal(monkeypatch):
    """Make stderr look like a terminal.

    Called in the test body: pytest re-installs its capture stream on
    sys.stderr after fixture setup.
    """
    monkeypatch.setattr(
        sys, "stderr", SimpleNamespace(isatty=lambda: True, write=lambda s: None)
    )


def run_in_threads(*fns):
    """Run each fn in its own thread; re-raise the first error."""
    errors = []

    def wrap(fn):
        try:
            fn()
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=wrap, args=(fn,)) for fn in fns]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    if errors:
        raise errors[0]


class TestMaxConcurrency:
    """max_concurrency caps one Model's in-flight requests across calls."""

    def test_cap_holds_across_two_threads(self, mock_litellm):
        """Two threads, 6 prompts each, cap 3: never more than 3 in flight."""
        fake = InFlight()
        mock_litellm.completion.side_effect = fake
        model = Model("gpt-4o", max_concurrency=3, show_progress=False)
        prompts = [f"Q{i}" for i in range(6)]
        results = {}
        run_in_threads(
            lambda: results.setdefault("a", model.generate_batch(prompts)),
            lambda: results.setdefault("b", model.generate_batch(prompts)),
        )
        assert fake.calls == 12
        assert fake.max == 3
        assert results["a"] == results["b"] == [[p] for p in prompts]
        assert len(model.history) == 12

    def test_no_cap_runs_both_calls_at_once(self, mock_litellm):
        """Without max_concurrency the same calls go far past 3 in flight."""
        fake = InFlight(delay=0.2)
        mock_litellm.completion.side_effect = fake
        model = Model("gpt-4o", show_progress=False)
        prompts = [f"Q{i}" for i in range(6)]
        run_in_threads(
            lambda: model.generate_batch(prompts),
            lambda: model.generate_batch(prompts),
        )
        assert fake.max > 3

    def test_fan_out_copies_count_against_the_cap(self, mock_litellm):
        """n=6 on a model without n support is 6 requests, at most 2 at once."""
        fake = InFlight()
        mock_litellm.completion.side_effect = fake
        model = Model("xai/grok-4", max_concurrency=2, show_progress=False)
        assert model.generate("Q", n=6) == ["Q"] * 6
        assert fake.calls == 6
        assert fake.max == 2

    def test_single_requests_share_the_cap(self, mock_litellm):
        """One-prompt calls from several threads also wait for a slot."""
        fake = InFlight()
        mock_litellm.completion.side_effect = fake
        model = Model("gpt-4o", max_concurrency=1)
        run_in_threads(*[lambda: model.generate("Q")] * 4)
        assert fake.calls == 4
        assert fake.max == 1

    def test_thread_pool_no_bigger_than_the_cap(self, mock_litellm):
        """A call starts at most max_concurrency worker threads."""
        mock_litellm.completion.side_effect = InFlight(delay=0)
        model = Model("gpt-4o", max_concurrency=2, show_progress=False)
        with patch.object(
            model_module, "ThreadPoolExecutor", wraps=model_module.ThreadPoolExecutor
        ) as pool:
            model.generate_batch(["A", "B", "C"])
        assert pool.call_args.kwargs["max_workers"] == 2

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_cap_rejected(self, bad):
        """max_concurrency must be positive."""
        with pytest.raises(ValueError, match="max_concurrency"):
            Model("gpt-4o", max_concurrency=bad)


class TestCacheKeysAndCopies:
    """max_concurrency is not part of the cache key; Model stays picklable."""

    def test_cache_key_and_canonical_unchanged(self):
        """Capped and uncapped models have the same cache_key and canonical form."""
        plain = Model("gpt-4o", temperature=0.3)
        capped = Model("gpt-4o", temperature=0.3, max_concurrency=4)
        assert capped.cache_key() == plain.cache_key()
        assert canonical(capped) == canonical(plain)

    def test_op_cache_hit_across_caps(self):
        """An @op run with an uncapped model is a cache hit with a capped one."""
        calls = []

        @op
        def model_name(model):
            calls.append(1)
            return model.cache_key()[1]

        assert model_name(Model("gpt-4o")) == "gpt-4o"
        assert model_name(Model("gpt-4o", max_concurrency=2)) == "gpt-4o"
        assert len(calls) == 1
        assert trace()[-1].cache_hit

    @pytest.mark.parametrize(
        "clone", [lambda m: pickle.loads(pickle.dumps(m)), copy.copy, copy.deepcopy]
    )
    def test_clones_get_their_own_cap(self, mock_litellm, clone):
        """Pickle and copy keep max_concurrency with a fresh, working semaphore."""
        model = Model("gpt-4o", max_concurrency=2)
        cloned = clone(model)
        assert cloned._max_concurrency == 2
        assert cloned._semaphore is not model._semaphore
        assert cloned.cache_key() == model.cache_key()
        fake = InFlight(delay=0.02)
        mock_litellm.completion.side_effect = fake
        cloned.generate_batch(["A", "B", "C", "D"], show_progress=False)
        assert fake.max <= 2

    def test_uncapped_model_pickles(self):
        """A model without a cap round-trips with no semaphore."""
        restored = pickle.loads(pickle.dumps(Model("gpt-4o")))
        assert restored._semaphore is None


class TestReturnExceptions:
    """return_exceptions=True keeps finished results when some requests fail."""

    def test_failed_prompt_gets_its_exception(self, mock_litellm):
        """Q2 fails: its slot holds the error, Q1 and Q3 keep their results."""
        mock_litellm.completion.side_effect = InFlight(delay=0, fail={"Q2"})
        model = Model("gpt-4o", show_progress=False)
        with patch("pyforger.acquire.model._tracking.add_usage") as add_usage:
            results = model.generate_batch(["Q1", "Q2", "Q3"], return_exceptions=True)
        assert len(results) == 3
        assert results[0] == ["Q1"] and results[2] == ["Q3"]
        assert isinstance(results[1], RuntimeError)
        assert "504 for Q2" in str(results[1])
        assert model.history == [results[0], results[2]]
        assert add_usage.call_count == 2

    def test_single_request_failure_is_returned(self, mock_litellm):
        """One prompt, one request: the exception comes back in the list."""
        error = RuntimeError("gateway timeout")
        mock_litellm.completion.side_effect = error
        model = Model("gpt-4o")
        assert model.generate_batch(["Q"], return_exceptions=True) == [error]
        assert model.history == []

    def test_partial_fan_out_keeps_finished_copies(self, mock_litellm):
        """One of A's 3 fan-out copies fails: A gets 2 completions, no ValueError."""
        mock_litellm.completion.side_effect = InFlight(delay=0, fail={"A@1"})
        model = Model("xai/grok-4", max_concurrency=1)
        results = model.generate_batch(["A", "B"], n=3, return_exceptions=True)
        assert results == [["A", "A"], ["B", "B", "B"]]
        assert all(isinstance(r, GenerateResult) for r in results)
        assert results[0].usage["prompt_tokens"] == 20  # the 2 finished requests
        assert model.total_usage["prompt_tokens"] == 50

    def test_all_fan_out_copies_failed(self, mock_litellm):
        """Every copy of a prompt failed: the slot holds one of the exceptions."""
        mock_litellm.completion.side_effect = InFlight(delay=0, fail={"A"})
        model = Model("xai/grok-4", show_progress=False)
        results = model.generate_batch(["A", "B"], n=2, return_exceptions=True)
        assert isinstance(results[0], RuntimeError)
        assert results[1] == ["B", "B"]

    def test_wrong_count_on_a_successful_prompt_still_raises(self, mock_litellm):
        """Short even after the n=1 refill: a ValueError, even with the flag."""

        def fake_completion(messages, n=1, **kwargs):
            response = make_response("only one")
            if n == 1:
                response.choices = []
            return response

        mock_litellm.completion.side_effect = fake_completion
        with pytest.raises(ValueError, match="returned 1 completions"):
            Model("gpt-4o").generate_batch(["Q"], n=2, return_exceptions=True)

    def test_default_still_raises(self, mock_litellm):
        """Without the flag, the first failure is raised and history stays empty."""
        mock_litellm.completion.side_effect = InFlight(delay=0, fail={"Q2"})
        model = Model("gpt-4o", show_progress=False)
        with pytest.raises(RuntimeError, match="504 for Q2"):
            model.generate_batch(["Q1", "Q2"])
        assert model.history == []

    def test_raised_batch_still_records_finished_usage(self, mock_litellm):
        """Q2 fails without the flag: Q1's and Q3's spend reach add_usage."""
        mock_litellm.completion.side_effect = InFlight(delay=0, fail={"Q2"})
        model = Model("gpt-4o", show_progress=False)
        with patch("pyforger.acquire.model._tracking.add_usage") as add_usage:
            with pytest.raises(RuntimeError, match="504 for Q2"):
                model.generate_batch(["Q1", "Q2", "Q3"])
        assert add_usage.call_count == 2
        assert all(c.args[:2] == (10, 20) for c in add_usage.call_args_list)
        assert model.history == []

    def test_local_model_engine_error_fills_every_slot(self, monkeypatch):
        """A LocalModel engine error fills every slot: vLLM runs one call."""
        error = RuntimeError("CUDA out of memory")

        class FailingLLM:
            def __init__(self, **kwargs):
                pass

            def chat(self, *args, **kwargs):
                raise error

        module = types.ModuleType("vllm")
        module.LLM = FailingLLM
        module.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
        monkeypatch.setitem(sys.modules, "vllm", module)
        model = LocalModel("Qwen/Qwen3-4B")
        assert model.generate_batch(["a", "b"], return_exceptions=True) == [
            error,
            error,
        ]
        with pytest.raises(RuntimeError, match="out of memory"):
            model.generate_batch(["a", "b"])


class TestProgress:
    """Progress is on by default: a bar in a terminal, a log line otherwise."""

    def test_default_is_log_without_a_terminal(self, monkeypatch):
        """Under pytest stderr is captured, so the default mode is the log line."""
        monkeypatch.delenv("FORGER_PROGRESS", raising=False)
        assert _progress_mode(None) == "log"

    def test_default_is_bar_in_a_terminal(self, monkeypatch):
        """A terminal gets the tqdm bar."""
        fake_terminal(monkeypatch)
        monkeypatch.delenv("FORGER_PROGRESS", raising=False)
        assert _progress_mode(None) == "bar"

    @pytest.mark.parametrize("value", ["0", "false", "no", "NO"])
    def test_env_turns_it_off(self, monkeypatch, value):
        """FORGER_PROGRESS=0 turns the default off."""
        monkeypatch.setenv("FORGER_PROGRESS", value)
        assert _progress_mode(None) == "off"

    def test_explicit_setting_beats_env(self, monkeypatch):
        """show_progress=True/False on the Model or call wins over the env."""
        monkeypatch.setenv("FORGER_PROGRESS", "0")
        assert _progress_mode(True) == "log"
        monkeypatch.setenv("FORGER_PROGRESS", "1")
        assert _progress_mode(False) == "off"

    def test_model_default_reaches_the_sender(self, mock_litellm, monkeypatch):
        """Model() with no show_progress sends with progress on; env=0 turns it off."""
        mock_litellm.completion.side_effect = InFlight(delay=0)
        model = Model("gpt-4o")
        with patch(
            "pyforger.acquire.model._batch_completion_with_progress",
            wraps=model_module._batch_completion_with_progress,
        ) as send:
            monkeypatch.delenv("FORGER_PROGRESS", raising=False)
            model.generate_batch(["A", "B"])
            monkeypatch.setenv("FORGER_PROGRESS", "0")
            model.generate_batch(["A", "B"])
        assert [c.kwargs["progress"] for c in send.call_args_list] == ["log", "off"]

    def test_log_mode_logs_at_the_interval(self, mock_litellm, monkeypatch, caplog):
        """With the interval at 0, each finished request logs a line."""
        monkeypatch.setattr(model_module, "_PROGRESS_LOG_INTERVAL_S", 0.0)
        mock_litellm.completion.side_effect = InFlight(delay=0)
        with caplog.at_level(logging.INFO, logger="pyforger.acquire.model"):
            Model("gpt-4o", show_progress=True).generate_batch(["A", "B", "C"])
        lines = [r.getMessage() for r in caplog.records]
        assert len(lines) == 3
        assert lines[-1].startswith("LLM calls (gpt-4o): 3/3 done, ")

    def test_fast_batch_logs_nothing(self, mock_litellm, caplog):
        """A batch done within 30 seconds prints no progress line."""
        mock_litellm.completion.side_effect = InFlight(delay=0)
        with caplog.at_level(logging.INFO, logger="pyforger.acquire.model"):
            Model("gpt-4o", show_progress=True).generate_batch(["A", "B", "C"])
        assert caplog.records == []

    def test_bar_in_a_terminal(self, mock_litellm, monkeypatch):
        """In a terminal the batch goes through tqdm with the request total."""
        import tqdm

        fake_terminal(monkeypatch)

        bars = []

        def fake_tqdm(iterable, **kwargs):
            bars.append(kwargs)
            return iterable

        monkeypatch.setattr(tqdm, "tqdm", fake_tqdm)
        mock_litellm.completion.side_effect = InFlight(delay=0)
        Model("gpt-4o").generate_batch(["A", "B"])
        assert bars == [{"total": 2, "desc": "LLM calls (gpt-4o)", "leave": False}]

    def test_off_uses_no_bar(self, mock_litellm, monkeypatch):
        """show_progress=False skips tqdm even in a terminal."""
        import tqdm

        fake_terminal(monkeypatch)

        monkeypatch.setattr(tqdm, "tqdm", MagicMock(side_effect=AssertionError))
        mock_litellm.completion.side_effect = InFlight(delay=0)
        assert Model("gpt-4o", show_progress=False).generate_batch(["A", "B"]) == [
            ["A"],
            ["B"],
        ]


class TestOneBatchByDefault:
    """Ops send all prompts in one generate_batch call unless batch_size is set."""

    def test_generate_from_context_one_call(self):
        """25 items are one generate_batch call of 25 prompts."""
        items = [{"text": str(i)} for i in range(25)]
        model = RecordingMockModel(responses=["ok"] * 25)
        out = generate_from_context(items, "p: {text}", model)
        assert out == [{"text": "ok"}] * 25
        assert [len(batch) for batch in model.recorded_prompts] == [25]

    def test_generate_from_context_retries_only_failures(self):
        """Parse failures in the single call are retried in one smaller call."""
        items = [{"text": str(i)} for i in range(12)]
        responses = ['{"key": "ok"}'] * 12
        responses[3] = responses[7] = "bad json"
        model = RecordingMockModel(responses=responses)
        out = generate_from_context(
            items, "p: {text}", model, output_schema={"key": str}, max_retries=1
        )
        assert out == [{"key": "ok"}] * 12
        first, retry = model.recorded_prompts
        assert len(first) == 12
        assert retry == [first[3], first[7]]

    def test_generate_from_context_explicit_batch_size_chunks(self):
        """An int batch_size keeps the old chunking."""
        items = [{"text": str(i)} for i in range(25)]
        model = RecordingMockModel(responses=["ok"] * 25)
        generate_from_context(items, "p: {text}", model, batch_size=10)
        assert [len(batch) for batch in model.recorded_prompts] == [10, 10, 5]

    def test_self_taught_reasoner_one_call(self):
        """All items go out in one generate_batch call."""
        items = [{"q": str(i), "final_answer": "4"} for i in range(15)]
        model = RecordingMockModel(responses=[["ANSWER: 4", "ANSWER: 5"]] * 15)
        kept = self_taught_reasoner(items, "{q}", model, n=2)
        assert len(kept) == 15
        assert [len(batch) for batch in model.recorded_prompts] == [15]

    def test_sample_from_llm_asks_for_all_samples_at_once(self):
        """n=30 is one generate call with n=30."""
        model = MockModel(responses=[[f"s{i}" for i in range(30)]])
        with patch.object(model, "generate", wraps=model.generate) as generate:
            items = sample_from_LLM("Generate", model, n=30)
        assert len(items) == 30
        assert generate.call_count == 1
        assert generate.call_args.kwargs["n"] == 30


class Numbered:
    """Fake completion: the i-th completion for prompt "P" across requests is "P#i".

    Records each request's n. Request k for "P" raises when "P@k" is in `fail`.
    """

    def __init__(self, fail=()):
        """Start every prompt's numbering at 0."""
        self.fail = set(fail)
        self.lock = threading.Lock()
        self.next: dict[str, int] = {}
        self.requests: dict[str, int] = {}
        self.ns: list[int] = []

    def __call__(self, messages, n=1, **kwargs):
        """Fake litellm.completion."""
        text = messages[-1]["content"]
        with self.lock:
            self.ns.append(n)
            k = self.requests.get(text, 0)
            self.requests[text] = k + 1
            start = self.next.get(text, 0)
            self.next[text] = start + n
        if f"{text}@{k}" in self.fail:
            raise RuntimeError(f"504 for {text}@{k}")
        response = make_response(text)
        response.choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=f"{text}#{i}", reasoning_content=None)
            )
            for i in range(start, start + n)
        ]
        return response


class TestLargeN:
    """n above _MAX_N_PER_REQUEST is split into concurrent requests."""

    @pytest.mark.parametrize(
        "n, sizes",
        [
            (1, [1]),
            (16, [16]),
            (17, [9, 8]),
            (40, [14, 13, 13]),
            (100, [15] * 2 + [14] * 5),
        ],
    )
    def test_request_sizes(self, n, sizes):
        """Near-equal parts of at most 16 that add up to n."""
        assert model_module._request_sizes(n, fan_out=False) == sizes

    def test_n_40_is_three_requests_in_order(self, mock_litellm):
        """n=40 on gpt-4o sends n=14, 13, 13; completions keep their order."""
        fake = Numbered()
        mock_litellm.completion.side_effect = fake
        model = Model("gpt-4o", max_concurrency=1)
        result = model.generate("Q", n=40)
        assert result == [f"Q#{i}" for i in range(40)]
        assert fake.ns == [14, 13, 13]
        assert result.usage["prompt_tokens"] == 30  # 3 requests x 10
        assert len(model.history) == 1

    def test_several_prompts_keep_prompt_order(self, mock_litellm):
        """Two prompts with n=20 get two requests each, merged per prompt."""
        fake = Numbered()
        mock_litellm.completion.side_effect = fake
        results = Model("gpt-4o", show_progress=False).generate_batch(["A", "B"], n=20)
        assert [sorted(r, key=lambda c: int(c.split("#")[1])) for r in results] == [
            [f"A#{i}" for i in range(20)],
            [f"B#{i}" for i in range(20)],
        ]
        assert sorted(fake.ns) == [10, 10, 10, 10]

    def test_failed_part_is_partial_with_return_exceptions(self, mock_litellm):
        """One of three parts fails: the prompt keeps the 27 that arrived."""
        mock_litellm.completion.side_effect = Numbered(fail={"Q@0"})
        model = Model("gpt-4o", max_concurrency=1)
        [result] = model.generate_batch(["Q"], n=40, return_exceptions=True)
        assert result == [f"Q#{i}" for i in range(14, 40)]

    def test_sample_from_llm_n_100(self, mock_litellm):
        """sample_from_LLM n=100: one generate call, no request above n=16."""
        fake = Numbered()
        mock_litellm.completion.side_effect = fake
        model = Model("gpt-4o", show_progress=False)
        with patch.object(model, "generate", wraps=model.generate) as generate:
            items = sample_from_LLM("Generate", model, n=100)
        assert len(items) == 100
        assert generate.call_count == 1
        assert max(fake.ns) <= 16 and sum(fake.ns) == 100

    def test_self_taught_reasoner_large_n(self, mock_litellm):
        """n=32 per item: requests of 16; every completion becomes a candidate."""
        fake = Numbered()
        mock_litellm.completion.side_effect = fake
        items = [{"q": "1+1", "final_answer": "1"}, {"q": "2+2", "final_answer": "2"}]
        kept = self_taught_reasoner(
            items, "{q}", Model("gpt-4o", show_progress=False), n=32
        )
        # "1+1#1" extracts 1 and "2+2#2" extracts 2: one match per item.
        assert [row["response"] for row in kept] == ["1+1#1", "2+2#2"]
        assert sorted(fake.ns) == [16] * 4

    def test_generate_from_context_large_samples_per_item(self, mock_litellm):
        """samples_per_item=20: requests of 10, 20 outputs per item in order."""
        fake = Numbered()
        mock_litellm.completion.side_effect = fake
        out = generate_from_context(
            [{"t": "A"}, {"t": "B"}],
            "{t}",
            Model("gpt-4o", max_concurrency=1),
            samples_per_item=20,
        )
        assert [row["text"] for row in out] == [f"A#{i}" for i in range(20)] + [
            f"B#{i}" for i in range(20)
        ]
        assert fake.ns == [10, 10, 10, 10]


class TestInterrupt:
    """Ctrl-C (or any error in the caller) cancels the requests not yet sent."""

    @pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError])
    def test_queued_requests_are_cancelled(self, mock_litellm, monkeypatch, error):
        """Interrupted after 4 of 50: the queue is dropped, in-flight ones finish."""
        fake = InFlight(delay=0.05)
        mock_litellm.completion.side_effect = fake

        def interrupt_after_four(futures, total, desc):
            for i, future in enumerate(futures):
                if i == 4:
                    raise error("interrupted")
                yield future

        monkeypatch.setattr(model_module, "_log_progress", interrupt_after_four)
        model = Model("gpt-4o", max_concurrency=2, show_progress=True)
        start = time.monotonic()
        with patch("pyforger.acquire.model._tracking.add_usage") as add_usage:
            with pytest.raises(error):
                model.generate_batch([f"Q{i}" for i in range(50)])
            assert time.monotonic() - start < 1.0  # 50 x 0.05s / 2 would be 1.25s
            time.sleep(0.3)  # let the requests in flight finish
        assert fake.calls <= 8
        assert fake.current == 0
        # Every request that finished was recorded, also those after the interrupt.
        assert add_usage.call_count == fake.calls
        assert model.history == []
