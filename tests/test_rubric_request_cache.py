"""Rubric generation reuses the request cache on first attempts only.

e.g. a pipeline that retries a 128-row chunk without its one failing input
pays only for that input's retries, not for the 127 rubrics it already has.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from test_request_cache import Completions, drop_op_cache, gpt

from pyforger.acquire import convert_rubric_criteria, generate_custom_rubric_criteria


class ByItem(Completions):
    """Completions keyed on the message holding the item, not the last one.

    Rubric prompts send the item as JSON, then a format instruction that is
    the same for every input.
    """

    def __call__(self, messages, n=1, **kwargs):
        """Answer as Completions would for the item's message."""
        item = next(m for m in messages if m["content"] in self.script)
        return super().__call__([item], n=n, **kwargs)


@pytest.fixture
def fake():
    """Patch litellm with a ByItem fake; an unscripted message raises."""
    completions = ByItem()
    with patch("pyforger.acquire.model.litellm") as mock:
        mock.get_supported_openai_params.return_value = None
        mock.completion.side_effect = completions
        yield completions


def asked(text: str) -> str:
    """The user message rubric generation sends for input `text`."""
    return json.dumps({"input": text}, ensure_ascii=False)


def rubric(*pairs) -> str:
    """A valid rubric reply."""
    return json.dumps({"rubric": [list(p) for p in pairs]})


def test_chunk_without_failing_input_reuses_the_others(fake):
    """The good input is asked once, the bad one twice; [good] then asks nothing."""
    model = gpt()
    fake.script[asked("good")] = [rubric(["Good?", 10])]
    fake.script[asked("bad")] = ["not json", "still not json"]
    with pytest.raises(ValueError, match=r"input indices \[1\]"):
        generate_custom_rubric_criteria(["good", "bad"], model, max_retries=1)
    assert sorted(text for text, _ in fake.calls) == [
        asked("bad"),
        asked("bad"),
        asked("good"),
    ]
    fake.reset()
    assert generate_custom_rubric_criteria(["good"], model, max_retries=1) == [
        [("Good?", 10.0)]
    ]
    assert fake.calls == []


def test_retry_is_fresh_and_later_calls_reuse_its_reply(fake, tmp_path):
    """A retry gets a new reply, not the malformed one; a rerun reuses it."""
    model = gpt()
    fake.script[asked("x")] = ["not json", rubric(["Fixed?", 5])]
    assert generate_custom_rubric_criteria(["x"], model, max_retries=1) == [
        [("Fixed?", 5.0)]
    ]
    assert len(fake.calls) == 2
    fake.reset()
    drop_op_cache(tmp_path)  # only the request cache can answer now
    fake.script[asked("y")] = [rubric(["New?", 3])]
    got = generate_custom_rubric_criteria(["x", "y"], model, max_retries=1)
    assert got == [[("Fixed?", 5.0)], [("New?", 3.0)]]
    assert [text for text, _ in fake.calls] == [asked("y")]


def test_no_cache_still_asks_every_input(fake):
    """_no_cache=True on the op regenerates every rubric."""
    model = gpt()
    fake.script[asked("x")] = [rubric(["A?", 1]), rubric(["B?", 1])]
    generate_custom_rubric_criteria(["x"], model)
    assert generate_custom_rubric_criteria(["x"], model, _no_cache=True) == [
        [("B?", 1.0)]
    ]
    assert len(fake.calls) == 2


def test_conversion_reuses_criteria_converted_before(fake):
    """Converting [A] after [A, B] asks nothing new for A."""
    model = gpt()
    for name in ("A?", "B?"):
        fake.script[json.dumps({"criterion": name}, ensure_ascii=False)] = [
            json.dumps({"kind": "prompt", "source": name})
        ]
    convert_rubric_criteria([[("A?", 1.0)], [("B?", 1.0)]], model)
    assert len(fake.calls) == 2
    fake.reset()
    assert convert_rubric_criteria([[("A?", 1.0)]], model) == [
        [[{"prompt": "A?"}, 1.0]]
    ]
    assert fake.calls == []
