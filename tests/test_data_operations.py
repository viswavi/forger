"""Generation and evaluation mechanisms compose independently."""
from __future__ import annotations

import json

import pytest
from completion import completion_server

from pyforger import alter, filter, inspect
from pyforger.acquire import Model
from pyforger.measurements import Measurement


def deterministic_generate(item, sample_id):
    """Generate a plain value without an agent, model, or directory."""
    return item["value"] + int(sample_id)


def deterministic_evaluate(item):
    """Evaluate plain existing data without knowing its producer."""
    return Measurement(passed=item["value"] == 6, score=float(item["value"]))


def test_measure_and_filter_share_evaluation(tmp_path, monkeypatch):
    """Inspect retains all measurements; selection does not change the data."""
    monkeypatch.setenv("FORGER_CACHE", str(tmp_path / "cache"))
    rows = [{"value": 5}, {"value": 6}]
    assert [r.passed for r in inspect.measure(rows, deterministic_evaluate)] == [
        False,
        True,
    ]
    assert filter.by_evaluation(rows, deterministic_evaluate) == rows[1:]
    assert (
        filter.by_evaluation(rows, deterministic_evaluate, accept=lambda r: r.score > 0)
        == rows
    )


def test_evaluator_errors_are_not_failed_grades():
    """An evaluator bug is not evidence of incorrect generated data."""

    def broken(item):
        raise RuntimeError("evaluator unavailable")

    with pytest.raises(RuntimeError):
        inspect.measure([{"value": 6}], broken)


def test_llm_generation_with_deterministic_evaluation(tmp_path, monkeypatch):
    """Exercise real model transport and ordinary code grading separately."""
    monkeypatch.setenv("FORGER_CACHE", str(tmp_path / "cache"))
    with completion_server({"role": "assistant", "content": "6"}) as (url, requests):
        model = Model("openai/test", api_key="test", api_base=url)

        def generate(item, sample_id):
            return int(model.generate("Sum 1,2,3", n=1)[0])

        generated = alter.sample([{}], generate, sample_ids=["one"])
    rows = [{"value": generated[0]["samples"][0]["value"]}]
    assert filter.by_evaluation(rows, deterministic_evaluate) == rows
    assert len(requests) == 1


def test_deterministic_generation_with_llm_evaluation(tmp_path, monkeypatch):
    """LLM judging accepts existing samples independent of generation."""
    monkeypatch.setenv("FORGER_CACHE", str(tmp_path / "cache"))
    rows = alter.sample([{"value": 5}], deterministic_generate, sample_ids=["1"])
    with completion_server({"role": "assistant", "content": "true"}) as (url, _):
        model = Model("openai/test", api_key="test", api_base=url)

        def evaluate(item):
            grade = json.loads(model.generate(str(item["samples"]), n=1)[0])
            return Measurement(passed=grade)

        assert filter.by_evaluation(rows, evaluate) == rows


@pytest.mark.parametrize("field,value", [("passed", 1), ("score", float("nan"))])
def test_measurement_structural_validation(field, value):
    """Pydantic rejects malformed grades without producer-specific validators."""
    with pytest.raises(ValueError):
        Measurement(**{field: value})


def test_agent_generation_composes_with_independent_evaluation(tmp_path, monkeypatch):
    """The actual local backend supplies values to ordinary evaluators."""
    from pyforger.agent_backend import AgentSpec
    from pyforger.alter import dynamic_step
    from pyforger.local_backend import LocalBackend

    monkeypatch.setenv("FORGER_CACHE", str(tmp_path / "cache"))
    agent = AgentSpec(
        kind="oracle",
        solution=(
            "cat > program.py <<'EOF'\n"
            "def run(item): return [str(sum(item['values']))]\n"
            "EOF\n"
        ),
    )
    values = dynamic_step([{"values": [1, 2, 3]}], "Sum", agent, LocalBackend())
    rows = [{"value": int(values[0][0])}]
    assert filter.by_evaluation(rows, deterministic_evaluate) == rows


class CountingEvaluator:
    """A stateful external evaluator with stable functional identity."""

    def __init__(self):
        """Keep execution observations outside the cache identity."""
        self.calls = 0

    def cache_key(self):
        """Identify the grader independently of its call counter."""
        return "counting-evaluator-v1"

    def __call__(self, item):
        """Grade a plain row and count real executions."""
        self.calls += 1
        return Measurement(passed=item["value"] == 6)


def test_filter_cache_bypass_really_reruns_evaluator(tmp_path, monkeypatch):
    """A shared helper must not hide another cache below the public operation."""
    monkeypatch.setenv("FORGER_CACHE", str(tmp_path / "cache"))
    evaluator = CountingEvaluator()
    rows = [{"value": 6}]
    assert filter.by_evaluation(rows, evaluator) == rows
    assert filter.by_evaluation(rows, evaluator) == rows
    assert evaluator.calls == 1
    assert filter.by_evaluation.no_cache(rows, evaluator) == rows
    assert evaluator.calls == 2
