"""Tests for pyforger.evaluate.quality.

End-to-end pipeline tests using a ``FakeBackend`` (same pattern as
``tests/test_inference_inspect_bridge.py``) and a tiny synthetic HumanEval
problem so we don't need a model or the real dataset on disk. The pipeline
still runs real subprocesses through ``score_program`` — that is the contract
we want to validate, and the executor handles its own isolation.
"""
from __future__ import annotations

from typing import Any

import pytest

from pyforger.evaluate.quality import (
    QualityReport,
    _bootstrap_ci,
    estimate_quality,
)
from pyforger.inference.backends import SamplingConfig


PROMPT = 'def add(x, y):\n    """Return x + y."""\n'
TEST = (
    "def check(candidate):\n"
    "    assert candidate(1, 2) == 3\n"
    "    assert candidate(-1, 1) == 0\n"
)


def _eval_item(task_id: str = "T/1") -> dict:
    """One HumanEval-shaped validation item."""
    return {
        "task_id": task_id,
        "prompt": PROMPT,
        "canonical_solution": "    return x + y\n",
        "test": TEST,
        "entry_point": "add",
    }


def _demos() -> list[dict]:
    """Three throwaway demos; tests that need real solutions construct their own."""
    return [
        {"prompt": "demo prompt 1", "solution": "demo solution 1"},
        {"prompt": "demo prompt 2", "solution": "demo solution 2"},
        {"prompt": "demo prompt 3", "solution": "demo solution 3"},
    ]


class FakeBackend:
    """Returns canned completions in call order; records what it saw.

    Mirrors ``tests/test_inference_inspect_bridge.py``'s ``FakeBackend``.
    """

    def __init__(self, completions: list[str], *, identity: str = "fake-model"):
        self.model = identity
        self._completions = list(completions)
        self.calls: list[tuple[list[Any], SamplingConfig]] = []

    def cache_key(self) -> dict:
        # Include the completions list so different test fixtures don't
        # collide on @op's disk cache (the autouse conftest fixture
        # already isolates per test, but be defensive).
        return {"backend": "fake", "identity": self.model, "completions": self._completions}

    def generate(self, prompts: list[Any], sampling: SamplingConfig) -> list[str]:
        self.calls.append((prompts, sampling))
        if len(prompts) > len(self._completions):
            raise AssertionError(
                f"FakeBackend has {len(self._completions)} canned "
                f"completions but was asked to generate {len(prompts)}"
            )
        return self._completions[: len(prompts)]


def _identity_render(messages):
    """Renderer that flattens messages into a deterministic string."""
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


class TestBootstrapCI:
    """Tiny sanity checks on the percentile bootstrap."""

    def test_constant_values_yield_tight_ci(self):
        lo, hi = _bootstrap_ci([0.5] * 10, n_resamples=200, seed=0)
        assert lo == pytest.approx(0.5)
        assert hi == pytest.approx(0.5)

    def test_empty_returns_zero_zero(self):
        assert _bootstrap_ci([], seed=0) == (0.0, 0.0)

    def test_lo_le_hi(self):
        lo, hi = _bootstrap_ci([0.0, 1.0, 0.0, 1.0, 0.0], seed=0)
        assert lo <= hi


class TestEstimateQuality:
    """estimate_quality wires generation -> scoring -> aggregation correctly."""

    def test_perfect_baseline_zero_uplift(self):
        # 1 problem, 2 subsets, k=1. Total prompts: 1 (zero-shot) + 1*2 = 3.
        # All three completions are correct -> baseline=1.0, conditioned=1.0.
        completions = ["    return x + y\n"] * 3
        backend = FakeBackend(completions)
        report = estimate_quality(
            backend,
            _demos(),
            [_eval_item()],
            render=_identity_render,
            k=1,
            n_subsets=2,
            sampling=SamplingConfig(temperature=0.0, max_tokens=64),
            seed=0,
            timeout_s=5,
            allow_code_execution=True,
        )
        assert report.baseline_pass_at_1 == 1.0
        assert report.conditioned_pass_at_1 == 1.0
        assert report.metric == 0.0
        assert isinstance(report, QualityReport)

    def test_demos_lift_failing_baseline(self):
        # 1 problem, 2 subsets. Zero-shot is wrong (subtraction); both
        # few-shot subsets are right. Expect uplift = 1.0.
        completions = [
            "    return x - y\n",   # zero-shot, wrong
            "    return x + y\n",   # subset 0
            "    return x + y\n",   # subset 1
        ]
        backend = FakeBackend(completions)
        report = estimate_quality(
            backend,
            _demos(),
            [_eval_item()],
            render=_identity_render,
            k=1,
            n_subsets=2,
            seed=0,
            timeout_s=5,
            allow_code_execution=True,
        )
        assert report.baseline_pass_at_1 == 0.0
        assert report.conditioned_pass_at_1 == 1.0
        assert report.metric == pytest.approx(1.0)

    def test_partial_uplift_aggregates_correctly(self):
        # 2 problems, 2 subsets each. Layout produced by estimate_quality:
        #   [zero-shot p0, zero-shot p1, p0 s0, p0 s1, p1 s0, p1 s1]
        completions = [
            "    return x - y\n",   # p0 zero-shot: fail
            "    return x + y\n",   # p1 zero-shot: pass
            "    return x + y\n",   # p0 subset 0: pass
            "    return x - y\n",   # p0 subset 1: fail
            "    return x + y\n",   # p1 subset 0: pass
            "    return x + y\n",   # p1 subset 1: pass
        ]
        backend = FakeBackend(completions)
        report = estimate_quality(
            backend,
            _demos(),
            [_eval_item("T/1"), _eval_item("T/2")],
            render=_identity_render,
            k=1,
            n_subsets=2,
            seed=0,
            timeout_s=5,
            allow_code_execution=True,
        )
        # Per-item:
        #   p0: baseline=0, subsets=[1,0] -> conditioned=0.5, uplift=+0.5
        #   p1: baseline=1, subsets=[1,1] -> conditioned=1.0, uplift= 0.0
        assert report.baseline_pass_at_1 == 0.5
        assert report.conditioned_pass_at_1 == 0.75
        assert report.metric == pytest.approx(0.25)
        assert report.per_item[0]["uplift"] == pytest.approx(0.5)
        assert report.per_item[1]["uplift"] == pytest.approx(0.0)

    def test_empty_eval_data_raises(self):
        with pytest.raises(ValueError):
            estimate_quality(
                FakeBackend([]),
                _demos(),
                [],
                render=_identity_render,
                k=1,
                n_subsets=1,
                allow_code_execution=True,
            )

    def test_empty_demos_with_k_positive_raises(self):
        with pytest.raises(ValueError):
            estimate_quality(
                FakeBackend([]),
                [],
                [_eval_item()],
                render=_identity_render,
                k=2,
                n_subsets=1,
                allow_code_execution=True,
            )

    def test_k_zero_skips_few_shot_generation(self):
        # k=0 means no demos -> only the zero-shot prompt is generated.
        backend = FakeBackend(["    return x + y\n"])
        report = estimate_quality(
            backend,
            _demos(),
            [_eval_item()],
            render=_identity_render,
            k=0,
            n_subsets=5,  # ignored when k=0
            allow_code_execution=True,
        )
        # Backend only saw 1 prompt (the zero-shot one).
        assert len(backend.calls) == 1
        assert len(backend.calls[0][0]) == 1
        # With no demos, conditioned defaults to baseline so metric is 0.
        assert report.metric == 0.0
        assert report.baseline_pass_at_1 == 1.0

    def test_render_is_invoked_per_prompt(self):
        completions = ["    return x + y\n"] * 3
        rendered: list[list[dict]] = []

        def capturing_render(messages):
            rendered.append(list(messages))
            return _identity_render(messages)

        backend = FakeBackend(completions)
        estimate_quality(
            backend,
            _demos(),
            [_eval_item()],
            render=capturing_render,
            k=2,
            n_subsets=2,
            allow_code_execution=True,
        )
        # 1 zero-shot + 2 few-shot = 3 renders.
        assert len(rendered) == 3
        # First render is zero-shot: just the target user message.
        assert rendered[0] == [{"role": "user", "content": PROMPT}]
        # Subsequent renders include 2 demo turn-pairs + 1 target turn = 5.
        assert len(rendered[1]) == 5
        assert rendered[1][-1] == {"role": "user", "content": PROMPT}

    def test_report_config_is_self_describing(self):
        backend = FakeBackend(["    return x + y\n"] * 3)
        report = estimate_quality(
            backend,
            _demos(),
            [_eval_item()],
            render=_identity_render,
            k=1,
            n_subsets=2,
            seed=42,
            allow_code_execution=True,
        )
        cfg = report.config
        assert cfg["k"] == 1
        assert cfg["n_subsets"] == 2
        assert cfg["seed"] == 42
        assert cfg["n_val"] == 1
        assert cfg["n_demos_pool"] == 3
        assert cfg["model"] == "fake-model"

    def test_report_serializes_to_dict(self):
        backend = FakeBackend(["    return x + y\n"] * 3)
        report = estimate_quality(
            backend,
            _demos(),
            [_eval_item()],
            render=_identity_render,
            k=1,
            n_subsets=2,
            allow_code_execution=True,
        )
        d = report.to_dict()
        assert set(d.keys()) >= {
            "metric",
            "baseline_pass_at_1",
            "conditioned_pass_at_1",
            "bootstrap_ci",
            "per_item",
            "config",
        }
