"""eval_tree's capability phrases: capped, delimited, and re-asked when not a phrase."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from pyforger.acquire.model import GenerateResult, Model  # noqa: E402
from pyforger.inspect.eval_tree import (  # noqa: E402
    _PHRASE_MAX_TOKENS,
    _PHRASE_RETRY_MAX_TOKENS,
    eval_tree,
)

CODE = "```python\nprint(sum(map(int, input().split())))\n```"


class Annotator(Model):
    """Answers each problem with `script[problem]` in turn, else "summing numbers"."""

    def __init__(self, script: dict[str, list[str]]):
        """Scripted answers per problem text."""
        super().__init__(model_id="test/annotator")
        self.script = script
        self.calls: list[dict[str, Any]] = []

    def cache_key(self) -> tuple:
        """Stable key for @op caching."""
        return ("Annotator",)

    def generate_batch(self, prompts, *, max_tokens=None, cache_salt=None, **kwargs):
        """One scripted completion per prompt."""
        out = []
        for prompt in prompts:
            user = prompt[-1]["content"]
            self.calls.append(
                {"user": user, "max_tokens": max_tokens, "salt": cache_salt}
            )
            key = next((k for k in self.script if k in user), None)
            text = (
                self.script[key].pop(0)
                if key and self.script[key]
                else "summing numbers"
            )
            out.append(
                GenerateResult(
                    [text], usage={}, reasoning=[None], model_id="m", raw=None
                )
            )
        return out


def rows(n: int = 4) -> list[dict]:
    """Scored problems ending in a solver instruction, as eval prompts do."""
    return [
        {
            "q": f"Problem {i}: add the numbers.\nWrite your solution in a code block.",
            "ok": i % 2,
        }
        for i in range(n)
    ]


def embed(texts: list[str]) -> list[list[float]]:
    """A fixed 2-d embedding, so no model download."""
    return [[float(i % 2), 1.0] for i, _ in enumerate(texts)]


def test_code_answer_is_asked_again_with_capped_tokens():
    """A solution instead of a phrase is re-asked; the problem is in <problem> tags."""
    model = Annotator({"Problem 0": [CODE, "parsing integers"]})
    report = eval_tree(
        rows(), input_field="q", score_field="ok", model=model, embed_fn=embed
    )
    assert report["capabilities"][0] == "parsing integers"
    first = model.calls[0]
    assert (
        first["user"].startswith("<problem>\n")
        and first["max_tokens"] == _PHRASE_MAX_TOKENS
    )
    assert [c["salt"] for c in model.calls if "Problem 0" in c["user"]] == [
        None,
        "retry 1",
    ]


def test_rows_without_a_phrase_are_left_out_of_the_tree(caplog):
    """After the retry, a row still without a phrase is not clustered or scored."""
    model = Annotator({"Problem 1": [CODE, "x" * 400]})
    report = eval_tree(
        rows(), input_field="q", score_field="ok", model=model, embed_fn=embed
    )
    assert report["capabilities"][1] is None and report["unannotated"] == [1]
    assert 1 not in report["tree"]["indices"] and report["tree"]["size"] == 3
    assert [c["max_tokens"] for c in model.calls if "Problem 1" in c["user"]] == [
        _PHRASE_MAX_TOKENS,
        _PHRASE_RETRY_MAX_TOKENS,
    ]
    assert "1 of 4 capability annotation responses were not a phrase" in caplog.text


def test_lowest_lists_subtrees_when_none_is_significant(caplog):
    """Too few rows for a significant weakness: lowest still ranks subtrees."""
    model = Annotator({})
    # Two groups of 4: 0/4 correct has an upper bound of 0.60, above threshold 0.5.
    data = [{"q": f"Problem {i}", "ok": int(i >= 4)} for i in range(8)]

    def two_groups(texts):
        return [[1.0, 0.0] if i < 4 else [0.0, 1.0] for i, _ in enumerate(texts)]

    report = eval_tree(
        data,
        input_field="q",
        score_field="ok",
        model=model,
        embed_fn=two_groups,
        min_ci_size=3,
    )
    assert report["weaknesses"] == []
    worst = report["lowest"][0]
    assert (worst["indices"], worst["accuracy"]) == ([0, 1, 2, 3], 0.0)
    assert "report['lowest'] lists the subtrees" in caplog.text


def test_eval_tree_wants_no_reasoning(monkeypatch):
    """eval_tree warns at "low" too; topic_based_clustering accepts it."""
    from pyforger.inspect.cluster import _warn_if_heavy_reasoning

    calls = []
    monkeypatch.setattr(
        "pyforger.inspect.cluster.logger.warning", lambda *a: calls.append(a)
    )
    _warn_if_heavy_reasoning(
        Model("gpt-4o", reasoning_effort="low"), "eval_tree", ("none",)
    )
    assert calls and "reasoning_effort='none'" in calls[0][0] % calls[0][1:]
