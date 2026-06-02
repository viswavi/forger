"""Tests for pyforger.inference.evaluate (and end-to-end integration)."""
from __future__ import annotations

from typing import Any

from pyforger.inference.backends import SamplingConfig
from pyforger.inference.benchmarks import gpqa as gpqa_module
from pyforger.inference.benchmarks.gpqa import GPQAExample
from pyforger.inference.evaluate import EvalResult, evaluate_gpqa
from pyforger.op import trace


class FakeBackend:
    """Test double satisfying the InferenceBackend Protocol.

    Returns a fixed list of completions in order, cycling if more prompts
    are requested than completions are supplied. Tracks the number of
    times ``.generate()`` is called so we can assert cache-hit behavior.
    """

    def __init__(
        self,
        completions: list[str],
        *,
        identity: str = "fake",
    ) -> None:
        self.model = identity
        self._completions = completions
        self._identity = identity
        self.call_count = 0

    def cache_key(self) -> dict:
        return {"backend": "fake", "identity": self._identity}

    def generate(self, prompts: list[str], sampling: SamplingConfig) -> list[str]:
        self.call_count += 1
        out = []
        for i, _ in enumerate(prompts):
            out.append(self._completions[i % len(self._completions)])
        return out


def _example(question: str, gold: str = "A") -> GPQAExample:
    return GPQAExample(
        question=question, choices=("w", "x", "y", "z"), gold=gold
    )


def _synthetic_rows(n: int) -> list[dict[str, Any]]:
    return [
        {
            "Question": f"Q{i}?",
            "Correct Answer": f"correct-{i}",
            "Incorrect Answer 1": f"w1-{i}",
            "Incorrect Answer 2": f"w2-{i}",
            "Incorrect Answer 3": f"w3-{i}",
        }
        for i in range(n)
    ]


def _last_record_for(op_name: str):
    """Most recent @op trace record matching ``op_name``."""
    matches = [r for r in trace() if r.op_name.endswith(op_name)]
    assert matches, f"no trace record for {op_name}"
    return matches[-1]


class TestEvalResult:
    """EvalResult is frozen so it pickles cleanly into the @op cache."""

    def test_construction(self):
        r = EvalResult(
            accuracy=0.5, n_total=4, n_correct=2, n_unparseable=0
        )
        assert r.accuracy == 0.5
        assert r.n_total == 4

    def test_equality(self):
        a = EvalResult(0.5, 2, 1, 0)
        b = EvalResult(0.5, 2, 1, 0)
        assert a == b


class TestAccuracyMath:
    """evaluate_gpqa's scoring logic, isolated from disk I/O."""

    def test_perfect_score(self, monkeypatch):
        rows = _synthetic_rows(4)
        monkeypatch.setattr(gpqa_module, "retrieve_from_hf", lambda **kw: rows)

        # The deterministic shuffle determines each example's gold letter;
        # peek at them so the fake backend can answer correctly.
        from pyforger.inference.benchmarks.gpqa import load_gpqa

        examples = load_gpqa(subset="gpqa_main", n=4)
        completions = [f"Answer: {ex.gold}" for ex in examples]

        fake = FakeBackend(completions, identity="perfect")
        result = evaluate_gpqa(fake, n=4)

        assert result == EvalResult(
            accuracy=1.0, n_total=4, n_correct=4, n_unparseable=0
        )

    def test_unparseable_counts_separately(self, monkeypatch):
        rows = _synthetic_rows(3)
        monkeypatch.setattr(gpqa_module, "retrieve_from_hf", lambda **kw: rows)

        # Three prompts, three unparseable completions.
        fake = FakeBackend(
            ["no letter at all !!", "no letter at all !!", "no letter at all !!"],
            identity="unparseable",
        )
        result = evaluate_gpqa(fake, n=3)

        assert result.n_total == 3
        assert result.n_correct == 0
        assert result.n_unparseable == 3
        assert result.accuracy == 0.0

    def test_mixed_correct_and_wrong(self, monkeypatch):
        rows = _synthetic_rows(4)
        monkeypatch.setattr(gpqa_module, "retrieve_from_hf", lambda **kw: rows)

        from pyforger.inference.benchmarks.gpqa import load_gpqa

        examples = load_gpqa(subset="gpqa_main", n=4)
        # Correct on the first two, wrong on the last two.
        wrong = {"A": "B", "B": "C", "C": "D", "D": "A"}
        completions = [
            f"Answer: {ex.gold}" if i < 2 else f"Answer: {wrong[ex.gold]}"
            for i, ex in enumerate(examples)
        ]

        fake = FakeBackend(completions, identity="mixed")
        result = evaluate_gpqa(fake, n=4)

        assert result.n_correct == 2
        assert result.n_total == 4
        assert result.n_unparseable == 0
        assert result.accuracy == 0.5


class TestCacheBehavior:
    """The @op decorator caches evaluate_gpqa keyed on inputs."""

    def test_second_call_is_cache_hit(self, monkeypatch):
        rows = _synthetic_rows(3)
        monkeypatch.setattr(gpqa_module, "retrieve_from_hf", lambda **kw: rows)

        fake = FakeBackend(["Answer: A"], identity="cache-hit")
        evaluate_gpqa(fake, n=3)
        first_calls = fake.call_count

        evaluate_gpqa(fake, n=3)
        # On the cache-hit path, the backend must not be re-invoked.
        assert fake.call_count == first_calls
        assert _last_record_for("evaluate_gpqa").cache_hit is True

    def test_few_shot_change_busts_cache(self, monkeypatch):
        rows = _synthetic_rows(3)
        monkeypatch.setattr(gpqa_module, "retrieve_from_hf", lambda **kw: rows)

        fake = FakeBackend(["Answer: A"], identity="few-shot-bust")
        evaluate_gpqa(fake, n=3)
        first_calls = fake.call_count

        shot = _example("shot?", gold="B")
        evaluate_gpqa(fake, n=3, few_shot=(shot,))
        # New cache key -> backend re-invoked, fresh trace record is a miss.
        assert fake.call_count == first_calls + 1
        assert _last_record_for("evaluate_gpqa").cache_hit is False

    def test_backend_identity_change_busts_cache(self, monkeypatch):
        rows = _synthetic_rows(3)
        monkeypatch.setattr(gpqa_module, "retrieve_from_hf", lambda **kw: rows)

        a = FakeBackend(["Answer: A"], identity="backend-A")
        b = FakeBackend(["Answer: A"], identity="backend-B")
        evaluate_gpqa(a, n=3)
        evaluate_gpqa(b, n=3)
        # Different backend.cache_key() -> the second call must hit the
        # fake (cache miss for evaluate_gpqa).
        assert b.call_count == 1
        assert _last_record_for("evaluate_gpqa").cache_hit is False


class TestEndToEndIntegration:
    """Wire retrieve_from_hf -> load_gpqa -> evaluate_gpqa end-to-end."""

    def test_full_stack_with_partial_correctness(self, monkeypatch):
        rows = _synthetic_rows(5)
        monkeypatch.setattr(gpqa_module, "retrieve_from_hf", lambda **kw: rows)

        from pyforger.inference.benchmarks.gpqa import load_gpqa

        examples = load_gpqa(subset="gpqa_main", n=5)
        # Answer correctly for the first three, wrong for the last two.
        wrong = {"A": "B", "B": "C", "C": "D", "D": "A"}
        completions = [
            f"Answer: {ex.gold}" if i < 3 else f"Answer: {wrong[ex.gold]}"
            for i, ex in enumerate(examples)
        ]
        fake = FakeBackend(completions, identity="e2e")

        result = evaluate_gpqa(fake, n=5)
        assert result == EvalResult(
            accuracy=0.6, n_total=5, n_correct=3, n_unparseable=0
        )

        # Second identical call: backend must not be re-entered.
        before = fake.call_count
        result2 = evaluate_gpqa(fake, n=5)
        assert result2 == result
        assert fake.call_count == before
        assert _last_record_for("evaluate_gpqa").cache_hit is True
