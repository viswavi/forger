"""Tests for LocalModel.target_logprobs and inspect.student_log_likelihood."""

from __future__ import annotations

import math
import multiprocessing
import sys
import threading
import types
from types import SimpleNamespace

import pytest

from pyforger import _tracking
from pyforger.acquire import _local_dp
from pyforger.acquire.model import LocalModel, TargetLogprobs
from pyforger.inspect import student_log_likelihood
from pyforger.inspect.likelihood import render_likelihood_report

_LN2 = math.log(2)


class _Tokenizer:
    """One token per character; the template joins contents and appends ">"."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        assert not tokenize and add_generation_prompt
        return "".join(m["content"] for m in messages) + ">"

    def __call__(self, text, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [ord(c) for c in text]}


class _SamplingParams:
    """Records ``vllm.SamplingParams`` kwargs (module level, so pipes can pickle it)."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def engines(monkeypatch):
    """A fake ``vllm``: every character costs 1 bit, except "z" (2 bits).

    ``chat`` answers every prompt with ``"aa"`` (1 bit per token).
    """
    created: list = []

    class FakeLLM:
        def __init__(self, **kwargs):
            self.calls: list[str] = []
            max_len = kwargs.get("max_model_len") or 10_000
            self.llm_engine = SimpleNamespace(
                model_config=SimpleNamespace(max_model_len=max_len)
            )
            created.append(self)

        def get_tokenizer(self):
            return _Tokenizer()

        def chat(self, messages, sampling_params, **kwargs):
            self.calls.append(f"chat@{sampling_params.kwargs['temperature']}")
            return [
                SimpleNamespace(
                    prompt_token_ids=[1],
                    outputs=[
                        SimpleNamespace(
                            text="aa", token_ids=[1, 2], finish_reason="stop"
                        )
                    ],
                )
                for _ in messages
            ]

        def generate(self, requests, params, use_tqdm=False):
            self.calls.append("generate")
            assert params.kwargs["prompt_logprobs"] == 0
            assert params.kwargs["max_tokens"] == 1
            outputs = []
            for request in requests:
                ids = request["prompt_token_ids"]
                logprobs = [None] + [
                    {t: SimpleNamespace(logprob=-_LN2 * (2 if t == ord("z") else 1))}
                    for t in ids[1:]
                ]
                outputs.append(SimpleNamespace(prompt_logprobs=logprobs))
            return outputs

    module = types.ModuleType("vllm")
    module.LLM = FakeLLM
    module.SamplingParams = _SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")  # one in-process engine
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    _tracking.clear_step_records()
    return created


class TestTargetLogprobs:
    """LocalModel.target_logprobs scores the target's tokens after the prompt."""

    def test_scores_only_the_target(self, engines):
        """The prompt "ab>" is context; "<think>zz</think>a" is 18 scored tokens."""
        [score] = LocalModel("m").target_logprobs(["ab"], ["<think>zz</think>a"])
        assert len(score.logprobs) == 18
        assert sum(-lp / _LN2 for lp in score.logprobs) == pytest.approx(20)
        assert score.reasoning_tokens == len("<think>zz</think>")
        assert not score.truncated

    def test_no_think_block_has_no_reasoning_tokens(self, engines):
        """A target without </think> counts every token as answer."""
        [score] = LocalModel("m").target_logprobs(["ab"], ["xyz"])
        assert score.reasoning_tokens == 0 and len(score.logprobs) == 3

    def test_target_cut_to_the_context(self, engines):
        """Only the tokens that fit (max_model_len - 1 in all) are scored."""
        model = LocalModel("m", max_model_len=10)
        [score] = model.target_logprobs(["ab"], ["<think>" + "x" * 20])
        assert score.truncated and len(score.logprobs) == 10 - 1 - 3

    def test_chat_prompts_and_order(self, engines):
        """Message-list prompts work, and scores come back in input order."""
        scores = LocalModel("m").target_logprobs(
            [[{"role": "user", "content": "q"}], "q"], ["x", "xyz"]
        )
        assert [len(s.logprobs) for s in scores] == [1, 3]

    def test_mismatched_lengths_raise(self, engines):
        """One target per prompt."""
        with pytest.raises(ValueError, match="one each"):
            LocalModel("m").target_logprobs(["a", "b"], ["x"])

    def test_data_parallel_deals_pairs_round_robin(self):
        """Pairs go to engine i % k and scores return in input order."""

        class Echo:
            def __init__(self):
                self.pairs = []

            def target_logprobs(self, pairs, **kwargs):
                self.pairs.append(pairs)
                return [p[1] for p in pairs]

        echoes = [Echo(), Echo()]
        engine = _local_dp.DataParallelEngine(echoes)
        pairs = [([], f"t{i}") for i in range(5)]
        assert engine.target_logprobs(pairs, use_tqdm=False) == [
            f"t{i}" for i in range(5)
        ]
        assert [len(e.pairs[0]) for e in echoes] == [3, 2]
        engine.close()

    def test_worker_answers_both_methods(self, engines):
        """A worker's ``serve`` loop dispatches chat and target_logprobs requests."""
        parent, child = multiprocessing.Pipe()
        worker = threading.Thread(target=_local_dp.serve, args=(child,))
        worker.start()
        parent.send({"model": "m"})
        assert parent.recv() == ("ready", None)
        params = _SamplingParams(temperature=0.5)
        parent.send(("chat", ([[{"role": "user", "content": "q"}]], params), {}))
        status, [output] = parent.recv()
        assert status == "ok" and output.outputs[0].text == "aa"
        parent.send(
            (
                "target_logprobs",
                ([([{"role": "user", "content": "q"}], "xz")],),
                {"chat_template_kwargs": {}},
            )
        )
        status, [score] = parent.recv()
        assert status == "ok" and len(score["logprobs"]) == 2
        parent.send(None)
        worker.join(timeout=10)
        assert not worker.is_alive()


def _rows(n: int) -> list[dict]:
    return [{"prompt": f"p{i}", "response": "<think>zz</think>a"} for i in range(n)]


class TestStudentLogLikelihood:
    """inspect.student_log_likelihood: bits, the own-response baseline, excess."""

    def test_bits_and_excess_against_own_rate(self, engines, capsys):
        """Own responses cost 1 bit/token: an 18-token, 20-bit target has 2 excess."""
        report = student_log_likelihood(_rows(4), LocalModel("m"), own_samples=2)
        assert [r.tokens for r in report.rows] == [18] * 4
        assert report.total_bits == pytest.approx(80)
        assert report.own_bits_per_token == pytest.approx(1.0)
        assert [r.row for r in report.own] == [0, 2]
        assert report.total_excess_bits == pytest.approx(8)
        assert report.rows[0].reasoning_tokens == 17
        assert "chat@1.0" in engines[0].calls  # own responses at temperature 1
        out = capsys.readouterr().out
        assert "student_log_likelihood" in out and "excess" in out

    def test_no_baseline(self, engines):
        """own_samples=0 samples nothing and leaves excess at 0."""
        report = student_log_likelihood(_rows(3), LocalModel("m"), own_samples=0)
        assert report.own == [] and report.own_bits_per_token is None
        assert report.total_excess_bits == 0
        assert engines[0].calls == ["generate"]

    def test_scores_and_samples_are_cached(self, engines):
        """A second call with the same rows and model loads both from the cache."""
        rows = _rows(3)
        student_log_likelihood(rows, LocalModel("m"), own_samples=1)
        calls = len(engines[0].calls)
        again = student_log_likelihood(rows, LocalModel("m"), own_samples=1)
        assert len(engines) == 1 and len(engines[0].calls) == calls
        assert again.total_excess_bits == pytest.approx(6)

    def test_field_names(self, engines):
        """prompt_field and target_field pick the fields to score."""
        rows = [{"q": "p", "gt": "xz"}]
        report = student_log_likelihood(
            rows, LocalModel("m"), prompt_field="q", target_field="gt", own_samples=0
        )
        assert report.rows[0].bits == pytest.approx(3)

    def test_model_without_scoring_raises(self):
        """A model that cannot score given text is refused up front."""
        with pytest.raises(TypeError, match="target_logprobs"):
            student_log_likelihood(_rows(1), object())

    def test_render(self, engines):
        """The plain-text report shows the table, the excess line and the top rows."""
        rows = _rows(3)
        report = student_log_likelihood(rows, LocalModel("m"), own_samples=3)
        text = render_likelihood_report(report, rows, top_k=2)
        assert "bits/token" in text and "student's own" in text
        assert "reasoning" in text and "answer" in text
        assert "most excess bits" in text and "row 0" in text
        assert "1.00 bits/token" in text


def test_target_logprobs_is_a_dataclass():
    """A TargetLogprobs is a plain value."""
    score = TargetLogprobs(logprobs=[-1.0], reasoning_tokens=0, truncated=False)
    assert score.logprobs == [-1.0]
