"""Tests for TinkerModel and the Tinker usage ledger, using fake tinker/transformers."""

from __future__ import annotations

import json
import math
import pickle
import sys
import threading
import time
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pyforger import _tracking, op, pipeline, step
from pyforger import tinker_usage as tinker_usage_module
from pyforger.acquire import (
    GenerateResult,
    PromptTooLongError,
    TinkerBudgetExceeded,
    TinkerModel,
    retrieve_from_web,
    sample_from_LLM,
    split_reasoning,
)
from pyforger.acquire import tinker_model as tinker_model_module
from pyforger.alter import generate_from_context, self_taught_reasoner
from pyforger.filter import by_llm_score, by_pass_at_k
from pyforger.tinker_usage import (
    TINKER_PRICES_PER_MILLION,
    TinkerUsageLedger,
    sample_tokens,
    tinker_cost_usd,
)

QWEN = "Qwen/Qwen3-8B"
IM_END, END_OF_TEXT = 151645, 151643
SPECIAL = {IM_END: "<|im_end|>", END_OF_TEXT: "<|endoftext|>"}
RATES = TINKER_PRICES_PER_MILLION[QWEN]


class FakeTokenizer:
    """Renders a Qwen-like chat template; one token per character."""

    def __init__(self):
        """Start with no recorded template calls."""
        self.template_calls: list[dict] = []
        self.eos_token_id = IM_END

    def apply_chat_template(self, messages, **kwargs):
        """Record the call and render ChatML text, ending in the assistant turn."""
        self.template_calls.append({"messages": messages, **kwargs})
        text = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )
        if kwargs.get("add_generation_prompt"):
            text += "<|im_start|>assistant\n"
            if kwargs.get("enable_thinking") is False:
                text += "<think>\n\n</think>\n\n"
        return text

    def encode(self, text, add_special_tokens=True):
        """One id per character (ord), so token counts equal text lengths."""
        assert add_special_tokens is False
        return [ord(c) for c in text]

    def decode(self, tokens, skip_special_tokens=False):
        """Inverse of encode; special ids decode to their names unless skipped."""
        out = []
        for t in tokens:
            if t in SPECIAL:
                if not skip_special_tokens:
                    out.append(SPECIAL[t])
            else:
                out.append(chr(t))
        return "".join(out)


class FakeError(Exception):
    """Base for the fake tinker exceptions."""


class World:
    """State shared by the fake tinker module: calls, responses, failures."""

    def __init__(self):
        """Default to Qwen3-8B with a 32k context and no failures."""
        self.calls: list[SimpleNamespace] = []
        self.services: list = []
        self.clients: list[dict] = []
        self.tokenizers: list[FakeTokenizer] = []
        self.path_base_model = QWEN
        self.max_context = 32768
        self.cache_hits = 0
        self.delay = 0.0
        self.failures: list[Exception] = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.respond = lambda call: [
            f"<think>t{j}</think>a{j}" for j in range(call.num_samples)
        ]
        # A sample's stop reason from its text; "length" ends without eos.
        self.stop_reason = lambda text: "stop"

    def text_of(self, call):
        """The rendered prompt text a call was made with."""
        return "".join(chr(t) for t in call.prompt.tokens)


@pytest.fixture
def world(monkeypatch):
    """Install fake ``tinker`` and ``transformers`` modules; return their state."""
    state = World()
    for name in (
        "FORGER_TINKER_USAGE_FILE",
        "FORGER_TINKER_USD_LIMIT",
        "TINKER_MAX_CONCURRENT_SAMPLES",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tinker_model_module, "_RETRY_DELAY_S", 0.0)

    class ModelInput:
        def __init__(self, tokens):
            self.tokens = tokens

        @classmethod
        def from_ints(cls, tokens):
            return cls(list(tokens))

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Future:
        def __init__(self, finish):
            self._finish = finish

        def result(self):
            return self._finish()

    class SamplingClient:
        def __init__(self, base_model):
            self.base_model = base_model

        def get_base_model(self):
            return self.base_model

        def sample(self, prompt, num_samples, sampling_params):
            call = SimpleNamespace(
                prompt=prompt,
                num_samples=num_samples,
                params=sampling_params.kwargs,
                client=self,
            )
            with state.lock:
                state.calls.append(call)
                state.active += 1
                state.max_active = max(state.max_active, state.active)

            def finish():
                time.sleep(state.delay)
                with state.lock:
                    state.active -= 1
                    failure = state.failures.pop(0) if state.failures else None
                if failure is not None:
                    raise failure
                texts = state.respond(call)
                reasons = [state.stop_reason(text) for text in texts]
                return SimpleNamespace(
                    sequences=[
                        SimpleNamespace(
                            tokens=[ord(c) for c in text]
                            + ([IM_END] if reason == "stop" else []),
                            stop_reason=reason,
                        )
                        for text, reason in zip(texts, reasons)
                    ],
                    prompt_cache_hit_tokens=state.cache_hits,
                )

            return Future(finish)

    class ServiceClient:
        def __init__(self, user_metadata=None):
            self.user_metadata = user_metadata
            self.holder = SimpleNamespace(_session_id="session-1")
            state.services.append(self)

        def create_sampling_client(self, model_path=None, base_model=None):
            state.clients.append({"model_path": model_path, "base_model": base_model})
            return SamplingClient(base_model or state.path_base_model)

        def get_server_capabilities(self):
            return SimpleNamespace(
                supported_models=[
                    SimpleNamespace(
                        model_name=QWEN, max_context_length=state.max_context
                    ),
                    SimpleNamespace(model_name="Other/Model", max_context_length=4096),
                ]
            )

    tinker = types.ModuleType("tinker")
    tinker.ServiceClient = ServiceClient
    tinker.types = SimpleNamespace(ModelInput=ModelInput, SamplingParams=SamplingParams)
    for name in ("APIConnectionError", "RateLimitError", "InternalServerError"):
        setattr(tinker, name, type(name, (FakeError,), {}))
    tinker.BadRequestError = type("BadRequestError", (FakeError,), {})

    def from_pretrained(name):
        tokenizer = FakeTokenizer()
        tokenizer.name = name
        state.tokenizers.append(tokenizer)
        return tokenizer

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=from_pretrained)
    transformers.GenerationConfig = SimpleNamespace(
        from_pretrained=lambda name: SimpleNamespace(eos_token_id=[IM_END, END_OF_TEXT])
    )
    monkeypatch.setitem(sys.modules, "tinker", tinker)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    state.tinker = tinker
    return state


def expected_prompt(content: str, thinking: bool | None = None) -> str:
    """The ChatML text FakeTokenizer renders for one user message."""
    text = f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    return text + ("<think>\n\n</think>\n\n" if thinking is False else "")


class TestConstruction:
    """Clients and imports are deferred to the first generation call."""

    def test_construction_does_not_need_tinker(self, monkeypatch):
        """Constructing a TinkerModel works without tinker installed."""
        monkeypatch.setitem(sys.modules, "tinker", None)
        monkeypatch.delenv("FORGER_TINKER_USAGE_FILE", raising=False)
        monkeypatch.delenv("FORGER_TINKER_USD_LIMIT", raising=False)
        model = TinkerModel(QWEN)
        assert model.history == []
        assert model.generate_batch([]) == []
        with pytest.raises(ImportError, match=r"pyforger\[tinker\]"):
            model.generate("hi")

    def test_no_client_until_generation(self, world):
        """No service client is created by the constructor or an empty batch."""
        model = TinkerModel(QWEN)
        model.generate_batch([])
        assert world.services == []

    def test_base_model_vs_checkpoint_path(self, world):
        """A tinker:// path samples saved weights; its base model renders prompts."""
        TinkerModel(QWEN).generate("hi")
        TinkerModel("tinker://run-1/sampler_weights/final").generate("hi")
        assert world.clients == [
            {"model_path": None, "base_model": QWEN},
            {"model_path": "tinker://run-1/sampler_weights/final", "base_model": None},
        ]
        assert [t.name for t in world.tokenizers] == [QWEN, QWEN]

    def test_client_is_reused(self, world):
        """Later calls reuse the first call's clients and tokenizer."""
        model = TinkerModel(QWEN)
        model.generate("a")
        model.generate_batch(["b", "c"])
        assert len(world.services) == 1 and len(world.tokenizers) == 1

    def test_pickle_drops_clients(self, world):
        """A used model pickles; the copy reopens clients lazily."""
        model = TinkerModel(QWEN)
        model.generate("a")
        copy = pickle.loads(pickle.dumps(model))
        assert copy._runtime is None
        assert copy.generate("b") == ["<think>t0</think>a0"]
        assert len(world.services) == 2


class TestRendering:
    """Prompts render with the HF chat template, as the evaluator's TinkerChat does."""

    def test_string_prompt_renders_with_generation_prompt(self, world):
        """A string is a user turn; the template's thinking default is kept."""
        TinkerModel(QWEN).generate("What is 2+2?")
        (call,) = world.calls
        assert world.text_of(call) == expected_prompt("What is 2+2?")
        template = world.tokenizers[0].template_calls[0]
        assert template["tokenize"] is False
        assert template["add_generation_prompt"] is True
        assert "enable_thinking" not in template

    def test_chat_messages_pass_through(self, world):
        """A message list keeps its system turn."""
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ]
        TinkerModel(QWEN).generate(messages)
        assert world.tokenizers[0].template_calls[0]["messages"] == messages
        assert world.text_of(world.calls[0]).startswith(
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi"
        )

    def test_enable_thinking_and_template_kwargs(self, world):
        """enable_thinking is forwarded; chat_template_kwargs win over it."""
        TinkerModel(QWEN, enable_thinking=False).generate("x")
        TinkerModel(
            QWEN, enable_thinking=False, chat_template_kwargs={"enable_thinking": True}
        ).generate("x")
        assert world.text_of(world.calls[0]) == expected_prompt("x", thinking=False)
        assert world.tokenizers[1].template_calls[0]["enable_thinking"] is True

    def test_decodes_without_special_tokens_and_keeps_thinking_inline(self, world):
        """<think> stays in the text; reasoning metadata is None per choice."""
        result = TinkerModel(QWEN).generate("x", n=2)
        assert isinstance(result, GenerateResult)
        assert list(result) == ["<think>t0</think>a0", "<think>t1</think>a1"]
        assert result.reasoning == [None, None]
        assert split_reasoning(result[1]) == ("t1", "a1")


class TestRequests:
    """One request per prompt, with n samples sharing its prefill."""

    def test_n_is_num_samples_in_one_request(self, world):
        """n=4 for 3 prompts makes 3 requests of num_samples=4."""
        results = TinkerModel(QWEN).generate_batch(["a", "b", "c"], n=4)
        assert len(world.calls) == 3
        assert {c.num_samples for c in world.calls} == {4}
        assert [len(r) for r in results] == [4, 4, 4]
        prompts = sorted(world.text_of(c) for c in world.calls)
        assert prompts == [expected_prompt(p) for p in "abc"]

    def test_results_keep_input_order(self, world):
        """Results follow input order even when requests finish out of order."""

        def respond(call):
            digit = world.text_of(call).split("user\nq")[1][0]
            time.sleep((6 - int(digit)) * 0.005)  # later prompts finish first
            return [digit]

        world.respond = respond
        prompts = [f"q{i}" for i in range(6)]
        results = TinkerModel(QWEN, max_concurrent=3).generate_batch(prompts)
        assert [r[0] for r in results] == [str(i) for i in range(6)]

    def test_sampling_params(self, world):
        """Defaults, per-call overrides and eos stop ids reach SamplingParams."""
        model = TinkerModel(QWEN, max_tokens=500, seed=7)
        model.generate("x")
        model.generate("x", temperature=1.0, max_tokens=50)
        first, second = (c.params for c in world.calls)
        assert first == {
            "max_tokens": 500,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "stop": [IM_END, END_OF_TEXT],
            "seed": 7,
        }
        assert (second["temperature"], second["max_tokens"]) == (1.0, 50)

    def test_default_max_tokens_is_clamped_to_context(self, world):
        """Prompt + completion stay inside the context window."""
        world.max_context = 100
        prompt_len = len(expected_prompt("hello"))
        result = TinkerModel(QWEN).generate("hello")
        assert world.calls[0].params["max_tokens"] == 100 - prompt_len
        assert result.usage["max_tokens_clamped"] is True

    def test_short_request_is_not_clamped(self, world):
        """A max_tokens that fits is sent unchanged."""
        result = TinkerModel(QWEN, max_tokens=10).generate("hello")
        assert world.calls[0].params["max_tokens"] == 10
        assert result.usage["max_tokens_clamped"] is False

    def test_prompt_longer_than_context_fails_without_a_request(self, world):
        """A prompt that leaves no room to generate raises before sampling."""
        world.max_context = 10
        with pytest.raises(PromptTooLongError, match="context window"):
            TinkerModel(QWEN).generate("hello")
        assert world.calls == []

    def test_long_prompt_in_a_batch_raises_before_any_request(self, world):
        """Every prompt is checked first, so the short ones are not sampled in vain."""
        world.max_context = len(expected_prompt("x" * 20))
        with pytest.raises(PromptTooLongError, match="Prompt 1 has"):
            TinkerModel(QWEN).generate_batch(["hi", "x" * 40, "ok"])
        assert world.calls == []

    def test_long_prompt_fills_its_slot_with_return_exceptions(self, world):
        """With return_exceptions the error is that prompt's; the others are sampled."""
        world.max_context = len(expected_prompt("x" * 20))
        results = TinkerModel(QWEN).generate_batch(
            ["hi", "x" * 40, "ok"], return_exceptions=True
        )
        assert isinstance(results[1], PromptTooLongError)
        assert [type(r) for r in (results[0], results[2])] == [GenerateResult] * 2
        assert len(world.calls) == 2

    def test_stop_strings_join_eos_tokens_and_cut_text(self, world):
        """Stop strings are sent with the eos tokens as strings; text ends before."""
        world.respond = lambda call: ["answer###tail"]
        result = TinkerModel(QWEN).generate("x", stop=["###"])
        assert world.calls[0].params["stop"] == ["<|im_end|>", "<|endoftext|>", "###"]
        assert list(result) == ["answer"]

    def test_invalid_n(self, world):
        """The sample count n must be positive."""
        with pytest.raises(ValueError, match="n must be positive"):
            TinkerModel(QWEN).generate("x", n=0)


class TestUsage:
    """Token counts and dollar costs follow Tinker's billing."""

    def test_usage_and_cost(self, world):
        """The shared prompt is billed once; other samples at the cache rate."""
        world.cache_hits = 10
        model = TinkerModel(QWEN)
        result = model.generate("hello", n=2)
        prompt_len = len(expected_prompt("hello"))
        sampled = [len("<think>t0</think>a0") + 1, len("<think>t1</think>a1") + 1]
        billed = {
            "prefill": prompt_len - 10,
            "cached_prefill": 10 + prompt_len,
            "sample": sum(sampled),
        }
        usage = result.usage
        assert usage["prompt_tokens"] == prompt_len
        assert usage["completion_tokens"] == sum(sampled)
        assert usage["cached_prompt_tokens"] == 10
        assert usage["reasoning_tokens"] == 0
        assert usage["billed_tokens"] == billed
        assert usage["stop_reasons"] == ["stop", "stop"]
        expected_cost = (
            billed["prefill"] * 0.195
            + billed["cached_prefill"] * 0.039
            + billed["sample"] * 0.60
        ) / 1e6
        assert usage["cost_usd"] == pytest.approx(expected_cost)
        # Qwen has no token-unit weight, as for LocalModel.
        assert math.isnan(usage["token_units"])
        model.generate("hello")
        assert len(model.history) == 2
        assert model.total_cost == pytest.approx(
            expected_cost + model.history[1].usage["cost_usd"]
        )
        assert model.total_usage["prompt_tokens"] == 2 * prompt_len

    def test_prices_override(self, world):
        """prices= replaces the built-in rate card for cost_usd."""
        prices = {"prefill": 1.0, "cached_prefill": 0.0, "sample": 0.0}
        result = TinkerModel(QWEN, prices=prices).generate("hi")
        assert result.usage["cost_usd"] == pytest.approx(
            len(expected_prompt("hi")) / 1e6
        )
        with pytest.raises(ValueError, match="missing"):
            TinkerModel(QWEN, prices={"prefill": 1.0})

    def test_unknown_prices_warn_and_cost_zero(self, world, monkeypatch):
        """A base model without prices reports zero cost and warns once."""
        monkeypatch.setattr(tinker_model_module, "_PRICE_WARNING_SEEN", set())
        world.path_base_model = "Other/Model"
        with pytest.warns(UserWarning, match="No Tinker prices"):
            result = TinkerModel("tinker://other/sampler_weights/x").generate("hi")
        assert result.usage["cost_usd"] == 0.0
        # The context window is looked up for the checkpoint's base model.
        prompt_len = len(expected_prompt("hi"))
        assert world.calls[0].params["max_tokens"] == 4096 - prompt_len


class TestCacheKey:
    """cache_key is stable, config-only, and separates models and settings."""

    def test_stable_and_client_free(self, world):
        """Equal configs share a key, before and after the clients open."""
        model = TinkerModel(QWEN)
        before = model.cache_key()
        model.generate("x")
        assert model.cache_key() == before == TinkerModel(QWEN).cache_key()
        assert all(
            isinstance(part, (str, int, float, dict, type(None))) for part in before
        )

    def test_distinguishes_model_and_sampling(self, world):
        """Checkpoint vs base, and each sampling setting, change the key."""
        base = TinkerModel(QWEN).cache_key()
        variants = [
            TinkerModel("tinker://run/sampler_weights/a"),
            TinkerModel("tinker://run/sampler_weights/b"),
            TinkerModel(QWEN, temperature=1.0),
            TinkerModel(QWEN, max_tokens=100),
            TinkerModel(QWEN, top_p=1.0),
            TinkerModel(QWEN, top_k=-1),
            TinkerModel(QWEN, enable_thinking=False),
            TinkerModel(QWEN, seed=1),
        ]
        keys = [base] + [m.cache_key() for m in variants]
        assert len(set(map(repr, keys))) == len(keys)

    def test_ignores_operational_settings(self, world):
        """Retries, concurrency, prices and budgets do not change outputs."""
        prices = {"prefill": 1.0, "cached_prefill": 1.0, "sample": 1.0}
        same = TinkerModel(
            QWEN,
            num_retries=5,
            max_concurrent=3,
            prices=prices,
            budget=RecordingBudget(),
            show_progress=True,
        )
        assert same.cache_key() == TinkerModel(QWEN).cache_key()

    def test_op_cache_hit_makes_no_request(self, world):
        """An @op called again with an equal model reads the disk cache."""

        @op
        def ask(model, question):
            return list(model.generate(question))

        assert ask(TinkerModel(QWEN), "q") == ask(TinkerModel(QWEN), "q")
        assert len(world.calls) == 1
        ask(TinkerModel("tinker://run/sampler_weights/a"), "q")
        assert len(world.calls) == 2


class RecordingBudget:
    """Duck-typed budget (like PostDataBench's TinkerBudget) that records calls."""

    def __init__(self, limit=float("inf")):
        """Track checks and records against an optional limit."""
        self.limit = limit
        self.spent = 0.0
        self.checks: list[float] = []
        self.records: list[tuple[dict, str | None]] = []

    def check(self, estimate_usd=0.0):
        """Refuse when the estimate would cross the limit."""
        self.checks.append(estimate_usd)
        if self.spent + estimate_usd > self.limit:
            raise TinkerBudgetExceeded("over")

    def record(self, tokens, session_id=None):
        """Charge billed tokens at Qwen3-8B rates."""
        self.records.append((tokens, session_id))
        self.spent += tinker_cost_usd(tokens, RATES)


class TestBudget:
    """Spend is charged to the environment ledger and to budget=, failing closed."""

    def configure(self, monkeypatch, path, limit):
        """Point the environment ledger at path with a dollar limit."""
        monkeypatch.setenv("FORGER_TINKER_USAGE_FILE", str(path))
        monkeypatch.setenv("FORGER_TINKER_USD_LIMIT", str(limit))

    def test_env_ledger_records_billed_tokens(self, world, monkeypatch, tmp_path):
        """Each request is charged with its billed tokens and session id."""
        ledger_path = tmp_path / "tinker_sft_usage.json"
        self.configure(monkeypatch, ledger_path, 150.0)
        model = TinkerModel(QWEN)
        first = model.generate("hello", n=3)
        model.generate("bye")
        state = json.loads(ledger_path.read_text())
        assert state["calls"] == 2
        assert state["model"] == QWEN and state["limit_usd"] == 150.0
        assert state["sessions"] == ["session-1"]
        assert state["tokens"]["prefill"] == (
            first.usage["billed_tokens"]["prefill"] + len(expected_prompt("bye"))
        )
        assert state["cost_usd"] == pytest.approx(model.total_cost)

    def test_estimate_over_limit_fails_closed(self, world, monkeypatch, tmp_path):
        """A call whose worst case would cross the limit is never sent."""
        self.configure(monkeypatch, tmp_path / "ledger.json", 0.001)
        model = TinkerModel(QWEN, max_tokens=10_000)  # $0.006 worst case
        with pytest.raises(TinkerBudgetExceeded, match="budget"):
            model.generate("hello")
        assert world.calls == []

    def test_exhausted_ledger_stops_later_calls(self, world, monkeypatch, tmp_path):
        """Once recorded spend reaches the limit, no further request starts."""
        path = tmp_path / "ledger.json"
        self.configure(monkeypatch, path, 0.01)
        TinkerUsageLedger(path, 0.01, model=QWEN).record({"sample": 16_667})
        with pytest.raises(TinkerBudgetExceeded):
            TinkerModel(QWEN, max_tokens=5).generate("x")
        assert world.calls == []

    def test_batch_stops_at_limit(self, world, monkeypatch, tmp_path):
        """Requests after the limit is reached fail; finished ones stay charged."""
        path = tmp_path / "ledger.json"
        prompt_cost = len(expected_prompt("x")) * 0.195 / 1e6
        # Room for one request's worst case (prompt + 20 sampled tokens), not two.
        limit = 1.5 * (prompt_cost + 20 * 0.6 / 1e6)
        self.configure(monkeypatch, path, limit)
        world.respond = lambda call: ["y" * 19]  # 20 tokens with <|im_end|>
        model = TinkerModel(QWEN, max_tokens=20, max_concurrent=1)
        with pytest.raises(TinkerBudgetExceeded):
            model.generate_batch(["x", "x", "x"])
        assert len(world.calls) == 1
        # history gets only returned results; the finished request is charged.
        assert model.history == []
        assert model.total_cost == pytest.approx(
            json.loads(path.read_text())["cost_usd"]
        )
        assert json.loads(path.read_text())["calls"] == 1

    def test_concurrent_requests_cannot_jointly_overshoot(
        self, world, monkeypatch, tmp_path
    ):
        """In-flight worst cases count against the limit, so 8 at once send 1."""
        path = tmp_path / "ledger.json"
        worst = (len(expected_prompt("x")) * 0.195 + 20 * 0.6) / 1e6
        limit = 1.5 * worst
        self.configure(monkeypatch, path, limit)
        world.delay = 0.05
        world.respond = lambda call: ["y" * 19]  # the full 20-token worst case
        model = TinkerModel(QWEN, max_tokens=20, max_concurrent=8)
        with pytest.raises(TinkerBudgetExceeded):
            model.generate_batch(["x"] * 8)
        assert len(world.calls) == 1
        assert json.loads(path.read_text())["cost_usd"] <= limit
        assert tinker_usage_module._RESERVED == {}

    def test_request_waits_for_in_flight_spend_then_runs(
        self, world, monkeypatch, tmp_path
    ):
        """A request that only fits once others are charged waits instead of failing."""
        path = tmp_path / "ledger.json"
        prompt_cost = len(expected_prompt("x")) * 0.195 / 1e6
        worst, actual = prompt_cost + 20 * 0.6 / 1e6, prompt_cost + 0.6 / 1e6
        limit = worst + 2 * actual + 1e-9
        assert 2 * worst > limit  # two worst cases never fit side by side
        self.configure(monkeypatch, path, limit)
        world.delay = 0.02
        world.respond = lambda call: [""]  # just <|im_end|>: 1 sampled token
        model = TinkerModel(QWEN, max_tokens=20, max_concurrent=4)
        assert model.generate_batch(["x"] * 3) == [[""]] * 3
        assert world.max_active == 1
        assert json.loads(path.read_text())["cost_usd"] == pytest.approx(3 * actual)

    def test_failed_request_releases_its_reservation(
        self, world, monkeypatch, tmp_path
    ):
        """An error mid-request does not leave its worst case held."""
        self.configure(monkeypatch, tmp_path / "ledger.json", 150.0)
        world.failures = [world.tinker.BadRequestError("bad")]
        model = TinkerModel(QWEN)
        with pytest.raises(world.tinker.BadRequestError):
            model.generate("x")
        assert tinker_usage_module._RESERVED == {}
        assert model.generate("x") == ["<think>t0</think>a0"]

    def test_half_configured_env_raises(self, world, monkeypatch, tmp_path):
        """Setting only one of the two variables is an error, not a free pass."""
        monkeypatch.setenv("FORGER_TINKER_USAGE_FILE", str(tmp_path / "x.json"))
        with pytest.raises(ValueError, match="FORGER_TINKER_USD_LIMIT"):
            TinkerModel(QWEN)

    def test_ledger_for_other_model_or_limit_fails_closed(
        self, world, monkeypatch, tmp_path
    ):
        """A ledger written for another model or limit refuses the run."""
        path = tmp_path / "ledger.json"
        TinkerUsageLedger(path, 150.0, model=QWEN).snapshot()
        self.configure(monkeypatch, path, 200.0)
        with pytest.raises(ValueError, match="ledger"):
            TinkerModel(QWEN).generate("x")
        assert world.calls == []

    def test_unpriced_model_with_ledger_fails_closed(
        self, world, monkeypatch, tmp_path
    ):
        """Without a rate card, a budgeted model refuses to sample."""
        self.configure(monkeypatch, tmp_path / "ledger.json", 150.0)
        world.path_base_model = "Other/Model"
        model = TinkerModel(
            "tinker://other/sampler_weights/x",
            prices={"prefill": 0.0, "cached_prefill": 0.0, "sample": 0.0},
        )
        with pytest.raises(ValueError, match="No Tinker prices"):
            model.generate("x")
        assert world.calls == []

    def test_explicit_budget_is_checked_and_charged(self, world):
        """budget= sees the worst-case estimate first, then the billed tokens."""
        budget = RecordingBudget()
        model = TinkerModel(QWEN, max_tokens=100, budget=budget)
        result = model.generate("x", n=2)
        prompt_len = len(expected_prompt("x"))
        estimate = sample_tokens(prompt_len, 2, 0, [100, 100])
        assert budget.checks == [pytest.approx(tinker_cost_usd(estimate, RATES))]
        assert budget.records == [(result.usage["billed_tokens"], "session-1")]

    def test_explicit_budget_refusal_sends_nothing(self, world):
        """A refusing budget stops the request."""
        model = TinkerModel(QWEN, budget=RecordingBudget(limit=0.0))
        with pytest.raises(TinkerBudgetExceeded):
            model.generate("x")
        assert world.calls == []

    def test_env_ledger_applies_alongside_explicit_budget(
        self, world, monkeypatch, tmp_path
    ):
        """Passing budget= does not bypass the environment ledger."""
        path = tmp_path / "ledger.json"
        self.configure(monkeypatch, path, 150.0)
        budget = RecordingBudget()
        TinkerModel(QWEN, budget=budget).generate("x")
        assert json.loads(path.read_text())["calls"] == 1
        assert len(budget.records) == 1

    def test_ledger_charges_builtin_rates_not_overrides(
        self, world, monkeypatch, tmp_path
    ):
        """prices= changes reported cost, never what the shared ledger charges."""
        path = tmp_path / "ledger.json"
        self.configure(monkeypatch, path, 150.0)
        zero = {"prefill": 0.0, "cached_prefill": 0.0, "sample": 0.0}
        result = TinkerModel(QWEN, prices=zero).generate("x")
        assert result.usage["cost_usd"] == 0.0
        assert json.loads(path.read_text())["cost_usd"] == pytest.approx(
            tinker_cost_usd(result.usage["billed_tokens"], RATES)
        )


class TestConcurrencyAndRetries:
    """In-flight requests are bounded; transient errors are retried."""

    def test_in_flight_requests_are_bounded(self, world):
        """No more than max_concurrent requests run at once."""
        world.delay = 0.02
        TinkerModel(QWEN, max_concurrent=2).generate_batch([str(i) for i in range(8)])
        assert len(world.calls) == 8
        assert world.max_active == 2

    def test_env_sets_default_concurrency(self, world, monkeypatch):
        """TINKER_MAX_CONCURRENT_SAMPLES is the default cap."""
        monkeypatch.setenv("TINKER_MAX_CONCURRENT_SAMPLES", "3")
        world.delay = 0.02
        TinkerModel(QWEN).generate_batch([str(i) for i in range(9)])
        assert world.max_active == 3

    def test_transient_error_is_retried(self, world):
        """A rate limit followed by success makes two requests."""
        world.failures = [world.tinker.RateLimitError("slow down")]
        assert TinkerModel(QWEN).generate("x") == ["<think>t0</think>a0"]
        assert len(world.calls) == 2

    def test_retries_are_bounded(self, world):
        """num_retries=1 allows two attempts, then raises."""
        world.failures = [world.tinker.APIConnectionError("down")] * 3
        with pytest.raises(world.tinker.APIConnectionError):
            TinkerModel(QWEN, num_retries=1).generate("x")
        assert len(world.calls) == 2

    def test_other_errors_are_not_retried(self, world):
        """A bad request fails at once; finished prompts are still charged."""
        world.failures = [world.tinker.BadRequestError("bad")]
        model = TinkerModel(QWEN, max_concurrent=1)
        with pytest.raises(world.tinker.BadRequestError):
            model.generate_batch(["a", "b"])
        assert len(world.calls) == 2
        assert model.history == []
        assert model.total_usage["prompt_tokens"] == len(expected_prompt("b"))


class TestOps:
    """TinkerModel works with every op that takes a Model."""

    def test_sample_from_llm(self, world):
        """sample_from_LLM gets n samples from one request per round."""
        rows = sample_from_LLM("Write a riddle.", TinkerModel(QWEN), n=3)
        assert [r["text"] for r in rows] == [
            f"<think>t{j}</think>a{j}" for j in range(3)
        ]
        assert [c.num_samples for c in world.calls] == [3]

    def test_generate_from_context(self, world):
        """generate_from_context fills the template per row."""
        world.respond = lambda call: ['{"answer": "4"}'] * call.num_samples
        rows = generate_from_context(
            [{"q": "2+2"}, {"q": "3+1"}],
            "Solve {q}.",
            TinkerModel(QWEN),
            output_schema={"answer": str},
        )
        assert [r["answer"] for r in rows] == ["4", "4"]
        assert sorted(world.text_of(c).split("\n")[1] for c in world.calls) == [
            "Solve 2+2.",
            "Solve 3+1.",
        ]

    def test_self_taught_reasoner(self, world):
        """The STaR op keeps correct samples with their inline thinking."""
        world.respond = lambda call: [
            "<think>add</think>ANSWER: 4",
            "<think>guess</think>ANSWER: 5",
        ]
        rows = self_taught_reasoner(
            [{"problem": "2+2", "final_answer": "4"}],
            "Solve {problem}.",
            TinkerModel(QWEN),
            n=2,
        )
        assert [(r["thinking"], r["student_final_answer"]) for r in rows] == [
            ("add", "4")
        ]
        assert [c.num_samples for c in world.calls] == [2]

    def test_long_prompt_is_a_row_error(self, world, monkeypatch):
        """An over-long row leaves its slot None; the op keeps the other rows."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        world.max_context = len(expected_prompt("x" * 20))

        @step
        def gen(rows: list[dict]) -> list[dict]:
            return generate_from_context(rows, "{q}", TinkerModel(QWEN), merge=True)

        out = gen([{"q": "hi"}, {"q": "x" * 40}])
        assert [r["q"] for r in out] == ["hi"]
        (warning,) = _tracking.step_records()["gen"].ops[0].warnings
        assert "1 prompt longer than the model's context window" in warning

    def test_retrieved_text_in_a_prompt_clears_the_reach_warning(
        self, world, monkeypatch
    ):
        """Retrieved text in a TinkerModel prompt counts, as for Model."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        words = " ".join(f"fact{j}" for j in range(60))
        page = {"text": f"The page says: {words}.", "id": "http://p", "url": "p"}

        @step
        def search(topics: list[dict]) -> list[dict]:
            return retrieve_from_web(queries=[t["topic"] for t in topics], n=1)

        def run(template: str) -> list[str]:
            @step
            def write(docs: list[dict]) -> list[dict]:
                rows = generate_from_context(docs, template, TinkerModel(QWEN))
                return [r for r in rows if r]

            @pipeline
            def build() -> list[dict]:
                return write(search([{"topic": "tides"}]))

            build()
            rec = _tracking.last_pipeline()
            assert rec is not None
            return [w for w in rec.warnings if "retrieve_from_web" in w]

        with patch(
            "pyforger.acquire.sources.WebRetrievalSource.fetch",
            side_effect=lambda *a, **k: [dict(page)],
        ):
            assert run("Ask about {query}.") != []  # the page never reached a prompt
            assert run("Ask about this page:\n{text}") == []

    def test_by_llm_score(self, world):
        """by_llm_score sends num_judges n=1 requests per row."""
        world.respond = lambda call: ['{"score": 5}'] * call.num_samples
        kept = by_llm_score(
            [{"q": "a", "r": "b"}],
            model=TinkerModel(QWEN, enable_thinking=False),
            input_field="q",
            output_field="r",
            num_judges=3,
        )
        assert kept == [{"q": "a", "r": "b"}]
        assert [c.num_samples for c in world.calls] == [1, 1, 1]


class TestLedger:
    """The ledger matches PostDataBench's TinkerBudget semantics."""

    def test_check_and_record(self, tmp_path):
        """Spend accumulates; a call that would cross the limit is refused."""
        ledger = TinkerUsageLedger(tmp_path / "l.json", 1.0, model=QWEN, name="sft")
        assert ledger.snapshot()["remaining_usd"] == 1.0
        state = ledger.record({"sample": 1_000_000}, "s1")
        assert state["cost_usd"] == pytest.approx(0.6)
        assert state["tokens"] == {
            "prefill": 0,
            "cached_prefill": 0,
            "sample": 1_000_000,
            "train": 0,
        }
        ledger.check(0.4)
        with pytest.raises(TinkerBudgetExceeded, match="sft budget"):
            ledger.check(0.41)

    def test_rejects_bad_limits(self, tmp_path):
        """Limits must be positive and finite; unknown models have no prices."""
        for limit in (0, -1, float("inf"), float("nan"), True):
            with pytest.raises(ValueError):
                TinkerUsageLedger(tmp_path / "l.json", limit, model=QWEN)
        with pytest.raises(ValueError, match="No Tinker prices"):
            TinkerUsageLedger(tmp_path / "l.json", 1.0, model="Other/Model")

    def test_corrupt_ledger_fails_closed(self, tmp_path):
        """A damaged cost refuses every check."""
        path = tmp_path / "l.json"
        ledger = TinkerUsageLedger(path, 1.0, model=QWEN)
        ledger.snapshot()
        state = json.loads(path.read_text())
        state["cost_usd"] = -1
        path.write_text(json.dumps(state))
        with pytest.raises(ValueError, match="Invalid"):
            ledger.check()


class TestModelContract:
    """TinkerModel follows Model's request cache, error, and accounting contract."""

    def test_request_cache_samples_only_what_is_missing(self, world):
        """More samples, or a new row, ask Tinker only for the new samples."""

        @op(name="test.tinker_ask", samples_arg="n")
        def ask(rows, model, n=2):
            prompts = [row["q"] for row in rows]
            results = model.generate_batch(prompts, n=n, cache_rows=rows)
            return [list(r) for r in results]

        rows = [{"q": "a"}, {"q": "b"}]
        first = ask(rows, TinkerModel(QWEN))
        assert sorted(c.num_samples for c in world.calls) == [2, 2]
        world.calls.clear()
        more = ask(rows, TinkerModel(QWEN), n=3)
        assert [r[:2] for r in more] == first
        assert sorted(c.num_samples for c in world.calls) == [1, 1]
        world.calls.clear()
        ask(rows + [{"q": "c"}], TinkerModel(QWEN), n=3)
        assert [(world.text_of(c), c.num_samples) for c in world.calls] == [
            (expected_prompt("c"), 3)
        ]

    def test_fully_cached_call_opens_no_client(self, world):
        """When every sample is cached, no Tinker client or tokenizer is created."""

        @op(name="test.tinker_ask_rows")
        def ask(rows, model):
            results = model.generate_batch([r["q"] for r in rows], cache_rows=rows)
            return [list(r) for r in results]

        ask([{"q": "a"}, {"q": "b"}], TinkerModel(QWEN))
        services, calls = len(world.services), len(world.calls)
        # Other rows miss the op cache, but row b's sample is in the request cache.
        assert ask([{"q": "b"}], TinkerModel(QWEN)) == [["<think>t0</think>a0"]]
        assert (len(world.services), len(world.calls)) == (services, calls)

    def test_return_exceptions_keeps_finished_prompts(self, world):
        """A failed request fills its prompt's slot; the others keep their samples."""
        world.failures = [world.tinker.BadRequestError("bad")]
        model = TinkerModel(QWEN, max_concurrent=1)
        results = model.generate_batch(["a", "b"], return_exceptions=True)
        assert isinstance(results[0], world.tinker.BadRequestError)
        assert results[1] == ["<think>t0</think>a0"]
        assert model.history == [results[1]]

    def test_step_report_and_ledger_show_tinker_spend(self, world, monkeypatch):
        """Tinker spend reaches the step's cost and the spend ledger as it happens."""
        from pyforger import _spend, _tracking, pipeline, step
        from pyforger.filter import by_length

        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        model = TinkerModel(QWEN)

        @step
        def write(rows, model):
            out = generate_from_context(rows, "Say {q}.", model)
            return by_length(out, instruction_field="text", min_len=0)

        @step
        def keep(rows):
            return by_length(rows, instruction_field="text", min_len=0)

        @pipeline
        def main(rows, model):
            return keep(write(rows, model))

        main([{"q": "a"}, {"q": "b"}], model)
        rec = _tracking.step_records()["write"]
        assert rec.cost_usd > 0
        assert rec.cost_usd == pytest.approx(model.total_cost)
        path = _spend.ledger_path()
        records = [r for r in _spend.read(path) if r["model_id"] == QWEN]
        assert len(records) == 2
        assert sum(r["cost_usd"] for r in records) == pytest.approx(model.total_cost)
        assert all(r["token_units"] is None for r in records)  # Qwen has no weights

    def test_pipeline_cap_stops_tinker_requests(self, world, monkeypatch):
        """max_cost_usd stops queued Tinker requests once spend reaches it."""
        from pyforger import BudgetExceededError, pipeline, step

        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        model = TinkerModel(QWEN, max_concurrent=1)

        @step
        def write(rows, model):
            return generate_from_context(rows, "Say {q}.", model)

        @pipeline(max_cost_usd=1e-9)
        def main(rows, model):
            return write(rows, model)

        with pytest.raises(BudgetExceededError, match="max_cost_usd"):
            main([{"q": str(i)} for i in range(5)], model)
        assert len(world.calls) == 1  # the requests queued behind it never left

    def test_transient_tinker_error_is_a_none_slot(self, world):
        """generate_from_context: a lost connection after retries is None, not fatal."""
        error_type = type("APIConnectionError", (FakeError,), {"__module__": "tinker"})
        world.tinker.APIConnectionError = error_type
        world.failures = [error_type("down")]
        rows = generate_from_context(
            [{"q": "a"}, {"q": "b"}],
            "Say {q}.",
            TinkerModel(QWEN, num_retries=0, max_concurrent=1),
            max_retries=0,
        )
        assert rows.count(None) == 1

    def test_by_pass_at_k_evaluates_a_tinker_student(self, world):
        """The student's k samples are one Tinker request per row (one prefill)."""
        from pyforger.filter import by_pass_at_k

        world.respond = lambda call: [
            "<think>add</think>4" if j % 2 == 0 else "<think>guess</think>5"
            for j in range(call.num_samples)
        ]
        kept = by_pass_at_k(
            [{"problem": "2+2", "answer": "4"}],
            model=TinkerModel(QWEN),
            input_field="problem",
            ground_truth_field="answer",
            k=4,
            outcomes_field="grades",
        )
        assert kept[0]["grades"] == {"model": [True, False, True, False]}
        assert [c.num_samples for c in world.calls] == [4]


class TestReviewFixes:
    """Budget, concurrency, and interruption behaviour found in review."""

    def test_budget_refusal_with_cached_samples_is_raised(self, world):
        """A budget refusal is not a short result, even when some samples are cached."""

        @op(name="test.tinker_more", samples_arg="n")
        def ask(rows, model, n):
            results = model.generate_batch(
                [r["q"] for r in rows], n=n, cache_rows=rows, return_exceptions=True
            )
            return [r if isinstance(r, Exception) else list(r) for r in results]

        rows = [{"q": "a"}]
        ask(rows, TinkerModel(QWEN), n=2)
        refused = ask(rows, TinkerModel(QWEN, budget=RecordingBudget(limit=0.0)), n=4)
        assert isinstance(refused[0], TinkerBudgetExceeded)

    def test_concurrency_cap_is_shared_across_instances(self, world, monkeypatch):
        """Two TinkerModels with different max_concurrent share the process cap."""
        monkeypatch.setenv("TINKER_MAX_CONCURRENT_SAMPLES", "3")
        world.delay = 0.03
        models = [
            TinkerModel(QWEN, max_concurrent=2),
            TinkerModel(QWEN, max_concurrent=3),
        ]
        threads = [
            threading.Thread(
                target=m.generate_batch, args=([str(i) for i in range(6)],)
            )
            for m in models
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(world.calls) == 12
        assert world.max_active == 3

    def test_extra_budget_reserves_in_flight_worst_cases(self, world):
        """budget= counts requests in flight, so 8 at once cannot all pass one check."""
        worst = (len(expected_prompt("x")) * 0.195 + 20 * 0.6) / 1e6
        budget = RecordingBudget(limit=1.5 * worst)
        world.delay = 0.05
        world.respond = lambda call: ["y" * 19]  # the full 20-token worst case
        model = TinkerModel(QWEN, max_tokens=20, max_concurrent=8, budget=budget)
        with pytest.raises(TinkerBudgetExceeded):
            model.generate_batch(["x"] * 8)
        assert len(world.calls) == 1
        assert budget.spent <= budget.limit

    def test_failed_ledger_write_keeps_spend_and_samples(self, world, monkeypatch):
        """Billed samples reach the totals and request cache if a ledger write fails."""

        class BrokenBudget(RecordingBudget):
            def record(self, tokens, session_id=None):
                raise OSError("disk full")

        @op(name="test.tinker_paid")
        def ask(rows, model):
            results = model.generate_batch([r["q"] for r in rows], cache_rows=rows)
            return [list(r) for r in results]

        model = TinkerModel(QWEN, budget=BrokenBudget())
        with pytest.raises(OSError, match="disk full"):
            ask([{"q": "a"}], model)
        assert model.total_cost > 0
        calls = len(world.calls)
        assert ask([{"q": "a"}, {"q": "b"}], TinkerModel(QWEN))
        assert len(world.calls) == calls + 1  # only b is sampled; a was cached

    def test_interrupt_stops_workers_in_retry_backoff(self, world, monkeypatch):
        """After an interrupt, a worker sleeping before its retry does not send it."""
        monkeypatch.setattr(tinker_model_module, "_RETRY_DELAY_S", 0.3)
        world.failures = [world.tinker.RateLimitError("slow down")]
        model = TinkerModel(QWEN, max_concurrent=1)

        def interrupt(*args, **kwargs):
            time.sleep(0.05)  # let the worker fail once and start its backoff
            raise KeyboardInterrupt

        monkeypatch.setattr(tinker_model_module, "_watch", interrupt)
        monkeypatch.setattr(tinker_model_module, "_progress_mode", lambda flag: "log")
        with pytest.raises(KeyboardInterrupt):
            model.generate_batch(["a", "b"])
        time.sleep(0.5)  # past the backoff
        assert len(world.calls) == 1

    def test_tinker_and_local_usage_add_no_token_units(self, world):
        """A weighted family name in a Tinker or local id still adds no units."""
        from pyforger.acquire import LocalModel

        for model in (
            TinkerModel("deepseek-ai/DeepSeek-V4-Flash", prices=RATES),
            LocalModel("zai-org/GLM-5.3"),
        ):
            assert math.isnan(model._calculate_token_units(10, 10))
            assert math.isnan(model.total_token_units)


def _cut_mid_thought(text: str) -> str:
    """Stop at max_tokens when the sample never closes its <think> block."""
    return "length" if "</think>" not in text else "stop"


class TestTruncation:
    """Samples cut off at max_tokens are reported once, with Tinker's advice."""

    def test_pass_at_k_warning_counts_samples_cut_mid_thought(self, world, monkeypatch):
        """The warning counts every max_tokens stop and agrees with the note.

        A bare op's note is never shown, and the pipeline summary shows only
        warnings: one that left out the mid-thought cuts said "2 of 8" when 6
        of 8 hit max_tokens.
        """
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        world.respond = lambda call: [
            "<think>still thinking",
            "<think>still thinking",
            "<think>x</think>4 and",
            "<think>x</think>4",
        ]
        world.stop_reason = lambda text: "stop" if text.endswith("4") else "length"
        model = TinkerModel(QWEN, max_tokens=50)

        @step
        def hard(rows: list[dict]) -> list[dict]:
            return by_pass_at_k(
                rows,
                model=model,
                input_field="task",
                ground_truth_field="answer",
                k=4,
                min_rate=0.0,
                max_rate=1.0,
            )

        hard([{"task": "2+2", "answer": "4"}, {"task": "3+1", "answer": "4"}])
        assert sum(r.truncated for r in model.history) == 6
        call = _tracking.step_records()["hard"].ops[0]
        assert call.warnings == [
            "filter.by_pass_at_k: 6 of 8 completions from Qwen/Qwen3-8B stopped at "
            "max_tokens=50 (finish_reason=length; 4 of them mid-thought, <think> "
            "never closed), so their text is cut off; raise the TinkerModel's "
            "max_tokens"
        ]
        assert call.scores.note == (
            "4 of 8 samples truncated (<think> never closed), graded failed; "
            "raise max_tokens if that is too many"
        )

    def test_finish_reasons_follow_their_slots_through_the_cache(self, world):
        """With n > 1 and some samples cached, each text keeps its own reason."""

        @op(name="test.tinker_reasons", samples_arg="n")
        def ask(rows, model, n=2):
            results = model.generate_batch([r["q"] for r in rows], n=n, cache_rows=rows)
            return [list(zip(r, r.finish_reasons)) for r in results]

        world.respond = lambda call: [
            f"<think>t{j}" if j % 2 else f"<think>t{j}</think>a"
            for j in range(call.num_samples)
        ]
        world.stop_reason = _cut_mid_thought
        ask([{"q": "a"}], TinkerModel(QWEN), n=2)
        world.calls.clear()
        (pairs,) = ask([{"q": "a"}], TinkerModel(QWEN), n=4)
        assert [c.num_samples for c in world.calls] == [2]  # slots 2 and 3 are new
        assert pairs == [
            ("<think>t0</think>a", "stop"),
            ("<think>t1", "length"),
            ("<think>t0</think>a", "stop"),
            ("<think>t1", "length"),
        ]

    def test_answer_cut_after_thinking_is_still_warned(self, world, monkeypatch):
        """A cut-off answer is not in the note, so the warning counts it."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        world.respond = lambda call: ["<think>x</think>4 and"] * call.num_samples
        world.stop_reason = lambda text: "length"

        @step
        def hard(rows: list[dict]) -> list[dict]:
            return by_pass_at_k(
                rows,
                model=TinkerModel(QWEN, max_tokens=50),
                input_field="task",
                ground_truth_field="answer",
                k=2,
                min_rate=0.0,
                max_rate=1.0,
            )

        hard([{"task": "2+2", "answer": "4"}])
        call = _tracking.step_records()["hard"].ops[0]
        assert call.warnings == [
            "filter.by_pass_at_k: 2 of 2 completions from Qwen/Qwen3-8B stopped at "
            "max_tokens=50 (finish_reason=length), so their text is cut off; raise "
            "the TinkerModel's max_tokens"
        ]
        assert call.scores.note is None

    def test_judge_cut_off_mid_thought_gets_one_warning(self, world, monkeypatch):
        """The unscored-calls warning covers it; the max_tokens one leaves it out."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        world.respond = lambda call: ["<think>hmm"] * call.num_samples
        world.stop_reason = _cut_mid_thought

        @step
        def judge(rows: list[dict]) -> list[dict]:
            return by_llm_score(
                rows,
                model=TinkerModel(QWEN, max_tokens=50),
                input_field="q",
                output_field="r",
            )

        assert judge([{"q": "a", "r": "b"}, {"q": "c", "r": "d"}]) == []
        assert _tracking.step_records()["judge"].ops[0].warnings == [
            "filter.by_llm_score: 2 of 2 judge calls gave no score (2 stopped "
            "mid-thought (unclosed <think>: raise max_tokens)) → 2 rows unscored"
        ]

    def test_other_ops_still_warn_about_unclosed_think(self, world, monkeypatch):
        """Outside the judge, which reports it, a cut-off thought is warned."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        world.respond = lambda call: ["<think>hmm"] * call.num_samples
        world.stop_reason = _cut_mid_thought

        @step
        def gen(rows: list[dict]) -> list[dict]:
            return generate_from_context(
                rows, "{q}", TinkerModel(QWEN, max_tokens=50), merge=True
            )

        gen([{"q": "a"}])
        warnings = _tracking.step_records()["gen"].ops[0].warnings
        # output_schema=None keeps any text, so the row stays, cut off.
        assert warnings == [
            "alter.generate_from_context: 1 row kept with cut-off text, because 1 "
            "of 1 completions from Qwen/Qwen3-8B stopped at max_tokens=50 "
            "(finish_reason=length; 1 of them mid-thought, <think> never closed); "
            "raise the TinkerModel's max_tokens"
        ]
