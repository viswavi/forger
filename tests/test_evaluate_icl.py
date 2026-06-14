"""Tests for pyforger.evaluate.icl.

Covers:
- ``sample_subsets`` determinism (same seed -> same subsets) and basic
  validation of bounds.
- ``build_messages`` shape: zero-shot returns a single user turn; few-shot
  returns alternating user/assistant turns ending on the target user turn.
- ``extract_code`` handles plain bodies, fenced blocks, and re-stated
  signatures (the three shapes instruct models reliably produce).
"""
from __future__ import annotations

import pytest

from pyforger.evaluate.icl import build_messages, extract_code, sample_subsets


def _demos(n: int) -> list[dict]:
    """Tiny synthetic demo pool for tests."""
    return [{"prompt": f"q{i}", "solution": f"a{i}"} for i in range(n)]


class TestSampleSubsets:
    """sample_subsets is deterministic and respects k/n_subsets bounds."""

    def test_same_seed_yields_same_subsets(self):
        pool = _demos(10)
        a = sample_subsets(pool, k=3, n_subsets=5, seed=42)
        b = sample_subsets(pool, k=3, n_subsets=5, seed=42)
        assert a == b

    def test_different_seeds_yield_different_subsets(self):
        pool = _demos(10)
        a = sample_subsets(pool, k=3, n_subsets=5, seed=0)
        b = sample_subsets(pool, k=3, n_subsets=5, seed=1)
        assert a != b

    def test_returns_correct_shape(self):
        pool = _demos(8)
        out = sample_subsets(pool, k=4, n_subsets=3, seed=0)
        assert len(out) == 3
        assert all(len(subset) == 4 for subset in out)

    def test_within_subset_no_replacement(self):
        pool = _demos(10)
        out = sample_subsets(pool, k=5, n_subsets=4, seed=7)
        for subset in out:
            ids = [d["prompt"] for d in subset]
            assert len(set(ids)) == len(ids)

    def test_zero_subsets_returns_empty_list(self):
        out = sample_subsets(_demos(5), k=2, n_subsets=0, seed=0)
        assert out == []

    def test_zero_k_returns_empty_subsets(self):
        out = sample_subsets(_demos(5), k=0, n_subsets=3, seed=0)
        assert out == [[], [], []]

    def test_k_larger_than_pool_raises(self):
        with pytest.raises(ValueError):
            sample_subsets(_demos(3), k=10, n_subsets=2, seed=0)

    def test_negative_k_raises(self):
        with pytest.raises(ValueError):
            sample_subsets(_demos(3), k=-1, n_subsets=2, seed=0)

    def test_negative_n_subsets_raises(self):
        with pytest.raises(ValueError):
            sample_subsets(_demos(3), k=2, n_subsets=-1, seed=0)


class TestBuildMessages:
    """build_messages produces the expected multi-turn shape."""

    def test_zero_shot_is_single_user_turn(self):
        msgs = build_messages([], "solve me")
        assert msgs == [{"role": "user", "content": "solve me"}]

    def test_few_shot_alternates_user_assistant_then_target(self):
        demos = [
            {"prompt": "q1", "solution": "a1"},
            {"prompt": "q2", "solution": "a2"},
        ]
        msgs = build_messages(demos, "target")
        assert msgs == [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "target"},
        ]

    def test_demo_missing_prompt_raises_keyerror(self):
        with pytest.raises(KeyError) as exc:
            build_messages([{"solution": "a"}], "target")
        assert "prompt" in str(exc.value)

    def test_demo_missing_solution_raises_keyerror(self):
        with pytest.raises(KeyError) as exc:
            build_messages([{"prompt": "q"}], "target")
        assert "solution" in str(exc.value)


class TestExtractCode:
    """extract_code strips fences but otherwise returns content as-is."""

    def test_plain_body_returned_unchanged(self):
        body = "    return x + 1"
        assert extract_code("def f(x):\n", body) == body

    def test_fenced_python_block_is_extracted(self):
        completion = (
            "Sure! Here is the solution:\n"
            "```python\n"
            "    return x * 2\n"
            "```\n"
            "Hope that helps!"
        )
        out = extract_code("def f(x):\n", completion)
        assert out.strip() == "return x * 2"

    def test_fenced_block_without_language_is_extracted(self):
        completion = "```\n    return 7\n```"
        out = extract_code("def f():\n", completion)
        assert out.strip() == "return 7"

    def test_restated_signature_inside_fence_is_kept(self):
        # We rely on Python's def-shadowing rather than trying to strip the
        # re-stated signature ourselves; this preserves any helper
        # definitions that come before the target.
        completion = (
            "```python\n"
            "def add(x, y):\n"
            "    return x + y\n"
            "```"
        )
        out = extract_code("def add(x, y):\n", completion)
        assert "def add(x, y):" in out
        assert "return x + y" in out

    def test_helper_then_target_both_kept(self):
        completion = (
            "```python\n"
            "def _helper(x):\n"
            "    return x * 2\n"
            "\n"
            "def add(x, y):\n"
            "    return _helper(x) + y\n"
            "```"
        )
        out = extract_code("def add(x, y):\n", completion)
        assert "def _helper(x):" in out
        assert "return _helper(x) + y" in out

    def test_only_first_fence_is_taken(self):
        completion = (
            "```python\nfirst\n```\nthen prose\n```python\nsecond\n```"
        )
        out = extract_code("def f():\n", completion)
        assert "first" in out
        assert "second" not in out
