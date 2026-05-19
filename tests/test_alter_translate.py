"""Tests for pyforger.alter.translate.to_language and translate.to_code."""

import pytest

from pyforger.alter import translate


class FakeModel:
    """Minimal stand-in for pyforger.acquire.model.Model.

    Records every generate_batch call and hands back one canned response per
    prompt in the order prompts arrive.
    """

    def __init__(self, responses):
        """Store the list of canned responses to hand back."""
        self._responses = list(responses)
        self.batch_calls = []

    def generate_batch(self, prompts, *, n=1, **_):
        """Record the prompt list and pop one canned response per prompt."""
        self.batch_calls.append(list(prompts))
        out = []
        for _ in prompts:
            out.append([self._responses.pop(0)])
        return out


class BrokenModel:
    """Fake that returns the wrong number of responses, for error-path tests."""

    def __init__(self, responses):
        """Store responses to return regardless of how many prompts arrive."""
        self._responses = responses

    def generate_batch(self, prompts, *, n=1, **_):
        """Return a fixed list, ignoring the prompt count."""
        return [[r] for r in self._responses]


class EmptyChoicesModel:
    """Fake whose first response is an empty choices list."""

    def generate_batch(self, prompts, *, n=1, **_):
        """Return an empty choices list for every prompt."""
        return [[] for _ in prompts]


# Shared signature + _execute pipeline; parametrize plumbing tests over both.
FACTORIES = [
    pytest.param(translate.to_language, "Spanish", id="to_language"),
    pytest.param(translate.to_code, "Python", id="to_code"),
]


@pytest.mark.parametrize("factory,target_language", FACTORIES)
class TestArgumentValidation:
    """Both factories validate their arguments at call time."""

    def test_empty_fields_raises(self, factory, target_language):
        """Empty fields list is rejected at call time."""
        source = [{"instruction": "hi"}]
        model = FakeModel([])
        with pytest.raises(ValueError, match="fields"):
            factory(
                source,
                target_language=target_language,
                fields=[],
                model=model,
            )

    def test_target_fields_length_mismatch_raises(self, factory, target_language):
        """target_fields must be the same length as fields."""
        source = [{"instruction": "hi", "response": "hello"}]
        model = FakeModel([])
        with pytest.raises(ValueError, match="target_fields"):
            factory(
                source,
                target_language=target_language,
                fields=["instruction", "response"],
                target_fields=["instruction_out"],
                model=model,
            )

    def test_target_fields_equal_length_accepted(self, factory, target_language):
        """Matched lengths should construct without error."""
        source = []
        model = FakeModel([])
        factory(
            source,
            target_language=target_language,
            fields=["instruction", "response"],
            target_fields=["instruction_out", "response_out"],
            model=model,
        )


@pytest.mark.parametrize("factory,target_language", FACTORIES)
class TestSingleField:
    """Translating a single field, in place."""

    def test_overwrites_in_place(self, factory, target_language):
        """With no target_fields, the source field is replaced."""
        data = [{"instruction": "hello"}]
        source = data
        model = FakeModel(["out"])
        result = factory(
            source,
            target_language=target_language,
            fields=["instruction"],
            model=model,
        )
        assert result == [{"instruction": "out"}]

    def test_multiple_items(self, factory, target_language):
        """Each item in the input gets its own translated output."""
        data = [{"instruction": "hello"}, {"instruction": "goodbye"}]
        source = data
        model = FakeModel(["one", "two"])
        result = factory(
            source,
            target_language=target_language,
            fields=["instruction"],
            model=model,
        )
        assert result == [{"instruction": "one"}, {"instruction": "two"}]

    def test_other_keys_pass_through(self, factory, target_language):
        """Fields not in the translate list are copied through unchanged."""
        data = [{"instruction": "hello", "id": 42, "tag": "greeting"}]
        source = data
        model = FakeModel(["out"])
        result = factory(
            source,
            target_language=target_language,
            fields=["instruction"],
            model=model,
        )
        assert result == [{"instruction": "out", "id": 42, "tag": "greeting"}]

    def test_leading_trailing_whitespace_stripped(self, factory, target_language):
        """Model output is stripped before being written back."""
        data = [{"instruction": "hello"}]
        source = data
        model = FakeModel(["  out\n"])
        result = factory(
            source,
            target_language=target_language,
            fields=["instruction"],
            model=model,
        )
        assert result == [{"instruction": "out"}]


@pytest.mark.parametrize("factory,target_language", FACTORIES)
class TestMultiField:
    """Translating more than one field per item, one LLM call per field."""

    def test_both_fields_replaced(self, factory, target_language):
        """Multi-field request translates every named field."""
        data = [{"instruction": "hello", "response": "hi there"}]
        source = data
        model = FakeModel(["A", "B"])
        result = factory(
            source,
            target_language=target_language,
            fields=["instruction", "response"],
            model=model,
        )
        assert result == [{"instruction": "A", "response": "B"}]

    def test_one_call_per_field_per_item(self, factory, target_language):
        """N items x M fields => N*M prompts in a single generate_batch call."""
        data = [
            {"instruction": "a", "response": "b"},
            {"instruction": "c", "response": "d"},
        ]
        source = data
        model = FakeModel(["A", "B", "C", "D"])
        # We don't care about the result here, and instead we just examine internally
        # how many times the model was called.
        factory(
            source,
            target_language=target_language,
            fields=["instruction", "response"],
            model=model,
        )
        assert len(model.batch_calls) == 1
        assert len(model.batch_calls[0]) == 4

    def test_prompt_order_is_item_major(self, factory, target_language):
        """Prompts are laid out as item 0's fields, then item 1's, etc."""
        data = [
            {"instruction": "a", "response": "b"},
            {"instruction": "c", "response": "d"},
        ]
        source = data
        model = FakeModel(["A", "B", "C", "D"])
        # We don't care about the result here, and instead we just examine internally
        # what prompts the factory generated and sent to the model.
        factory(
            source,
            target_language=target_language,
            fields=["instruction", "response"],
            model=model,
        )
        batch = model.batch_calls[0]
        user_msgs_per_prompt = [
            [m for m in prompt if m["role"] == "user"] for prompt in batch
        ]
        assert all(len(msgs) == 1 for msgs in user_msgs_per_prompt)
        user_contents = [msgs[0]["content"] for msgs in user_msgs_per_prompt]
        assert user_contents == ["a", "b", "c", "d"]

    def test_results_written_back_in_order(self, factory, target_language):
        """Translations land on the correct (item, field) slot."""
        data = [
            {"instruction": "a", "response": "b"},
            {"instruction": "c", "response": "d"},
        ]
        source = data
        model = FakeModel(["A", "B", "C", "D"])
        result = factory(
            source,
            target_language=target_language,
            fields=["instruction", "response"],
            model=model,
        )
        assert result == [
            {"instruction": "A", "response": "B"},
            {"instruction": "C", "response": "D"},
        ]


@pytest.mark.parametrize("factory,target_language", FACTORIES)
class TestTargetFields:
    """target_fields preserves originals and writes to new keys."""

    def test_originals_preserved(self, factory, target_language):
        """Source fields are kept and outputs go to new keys."""
        data = [{"instruction": "hello", "response": "hi"}]
        source = data
        model = FakeModel(["A", "B"])
        result = factory(
            source,
            target_language=target_language,
            fields=["instruction", "response"],
            target_fields=["instruction_out", "response_out"],
            model=model,
        )
        assert result == [
            {
                "instruction": "hello",
                "response": "hi",
                "instruction_out": "A",
                "response_out": "B",
            }
        ]

    def test_different_target_name(self, factory, target_language):
        """Single target_fields entry works the same as the multi-field case."""
        data = [{"instruction": "hello"}]
        source = data
        model = FakeModel(["out"])
        result = factory(
            source,
            target_language=target_language,
            fields=["instruction"],
            target_fields=["instruction_out"],
            model=model,
        )
        assert result == [{"instruction": "hello", "instruction_out": "out"}]


@pytest.mark.parametrize("factory,target_language", FACTORIES)
class TestMissingSourceField:
    """Missing source fields raise KeyError via get_field, wrapped by the runner."""

    def test_missing_field_raises(self, factory, target_language):
        """An item missing a listed field surfaces a KeyError via get_field."""
        data = [{"instruction": "hi"}]
        model = FakeModel([])
        with pytest.raises(KeyError, match="missing required field 'response'"):
            factory(
                data,
                target_language=target_language,
                fields=["instruction", "response"],
                model=model,
            )


@pytest.mark.parametrize("factory,target_language", FACTORIES)
class TestModelErrors:
    """Model-side failure modes surface as loud errors with context."""

    def test_length_mismatch_too_few_raises(self, factory, target_language):
        """Model returning fewer responses than prompts fails loudly."""
        data = [{"instruction": "hi"}, {"instruction": "hello"}]
        model = BrokenModel(["out"])  # 1 response for 2 prompts
        with pytest.raises(ValueError, match="1 completions for 2 prompts"):
            factory(
                data,
                target_language=target_language,
                fields=["instruction"],
                model=model,
            )

    def test_length_mismatch_too_many_raises(self, factory, target_language):
        """Model returning more responses than prompts also fails loudly."""
        data = [{"instruction": "hi"}]
        model = BrokenModel(["out", "extra"])  # 2 responses for 1 prompt
        with pytest.raises(ValueError, match="2 completions for 1 prompts"):
            factory(
                data,
                target_language=target_language,
                fields=["instruction"],
                model=model,
            )

    def test_wrong_shape_raises(self, factory, target_language):
        """Model returning flat list[str] instead of list[list[str]] fails."""

        class FlatModel:
            """Returns list[str] instead of list[list[str]]."""

            def generate_batch(self, prompts, *, n=1, **_):
                """Return a string per prompt, not a list of strings."""
                return ["out" for _ in prompts]

        data = [{"instruction": "hi"}]
        with pytest.raises(ValueError, match="expected list of completions"):
            factory(
                data,
                target_language=target_language,
                fields=["instruction"],
                model=FlatModel(),
            )

    def test_no_completions_for_prompt_raises(self, factory, target_language):
        """A prompt that gets an empty choices list names item and field."""
        data = [{"instruction": "hi", "response": "hello"}]
        model = EmptyChoicesModel()
        with pytest.raises(ValueError, match="item 0 field 'instruction'"):
            factory(
                data,
                target_language=target_language,
                fields=["instruction", "response"],
                model=model,
            )


@pytest.mark.parametrize("factory,target_language", FACTORIES)
class TestPromptContent:
    """System prompt and user message contain what they should."""

    def test_system_prompt_includes_target_language(self, factory, target_language):
        """The target language string appears in the system message."""
        data = [{"instruction": "hello"}]
        source = data
        model = FakeModel(["out"])
        # We don't care about the result here, and instead we just examine internally
        # how many times the model was called and what prompts were given.
        factory(
            source,
            target_language=target_language,
            fields=["instruction"],
            model=model,
        )
        prompt = model.batch_calls[0][0]
        assert isinstance(prompt, list)
        system = next(m for m in prompt if m["role"] == "system")
        assert target_language in system["content"]

    def test_user_message_is_raw_field_value(self, factory, target_language):
        """The user message is the source field value, not a wrapped version."""
        data = [{"instruction": "what is 2+2?"}]
        source = data
        model = FakeModel(["out"])
        # We don't care about the result here, and instead we just examine internally
        # how many times the model was called and what prompts were given.
        factory(
            source,
            target_language=target_language,
            fields=["instruction"],
            model=model,
        )
        prompt = model.batch_calls[0][0]
        user = next(m for m in prompt if m["role"] == "user")
        assert user["content"] == "what is 2+2?"

    def test_each_prompt_has_one_field_value(self, factory, target_language):
        """Multi-field translation sends each field as its own prompt."""
        data = [{"instruction": "A", "response": "B"}]
        source = data
        model = FakeModel(["a", "b"])
        # We don't care about the result here, and instead we just examine internally
        # how many times the model was called and what prompts were given.
        factory(
            source,
            target_language=target_language,
            fields=["instruction", "response"],
            model=model,
        )
        batch = model.batch_calls[0]
        assert len(batch) == 2
        user_msgs_per_prompt = [[m for m in p if m["role"] == "user"] for p in batch]
        assert all(len(msgs) == 1 for msgs in user_msgs_per_prompt)
        user_contents = [msgs[0]["content"] for msgs in user_msgs_per_prompt]
        assert user_contents == ["A", "B"]


@pytest.mark.parametrize("factory,target_language", FACTORIES)
class TestEdgeCases:
    """Boundary conditions."""

    def test_empty_input_skips_llm_call(self, factory, target_language):
        """No data means no generate_batch call is made."""
        source = []
        model = FakeModel([])
        result = factory(
            source,
            target_language=target_language,
            fields=["instruction"],
            model=model,
        )
        assert result == []
        assert model.batch_calls == []

    def test_empty_string_field_value_passed_through(self, factory, target_language):
        """An empty string is still sent to the model and its output used."""
        data = [{"instruction": ""}]
        source = data
        model = FakeModel([""])
        result = factory(
            source,
            target_language=target_language,
            fields=["instruction"],
            model=model,
        )
        assert result == [{"instruction": ""}]

    def test_multiline_field_content(self, factory, target_language):
        """Multi-line text is passed through as one piece, no extra processing."""
        data = [{"instruction": "line one\nline two\nline three"}]
        source = data
        model = FakeModel(["a\nb\nc"])
        result = factory(
            source,
            target_language=target_language,
            fields=["instruction"],
            model=model,
        )
        assert result == [{"instruction": "a\nb\nc"}]

    def test_input_dicts_not_mutated(self, factory, target_language):
        """The alter step returns new dicts and a new list; source is untouched."""
        original = {"instruction": "hello", "id": 1}
        data = [original]
        source = data
        model = FakeModel(["out"])
        result = factory(
            source,
            target_language=target_language,
            fields=["instruction"],
            model=model,
        )
        assert original == {"instruction": "hello", "id": 1}
        assert result[0] is not original
        assert result is not data


class TestToLanguageSpecific:
    """Behavior unique to to_language."""

    def test_markdown_fences_in_content_preserved(self):
        """to_language must not strip code fences; only to_code does that."""
        data = [{"doc": "explain a function"}]
        source = data
        translated = (
            "aqui esta la funcion:\n"
            "```python\n"
            "def foo():\n"
            "    return 1\n"
            "```"
        )
        model = FakeModel([translated])
        result = translate.to_language(
            source,
            target_language="Spanish",
            fields=["doc"],
            model=model,
        )
        assert result == [{"doc": translated}]


class TestToCodeFenceStripping:
    """to_code strips outer markdown code fences the model adds unprompted."""

    def test_fenced_with_language_tag_stripped(self):
        """```python ... ``` wrapping is removed."""
        data = [{"prompt": "add"}]
        source = data
        model = FakeModel(["```python\ndef add(a, b):\n    return a + b\n```"])
        result = translate.to_code(
            source,
            target_language="Python",
            fields=["prompt"],
            model=model,
        )
        assert result == [{"prompt": "def add(a, b):\n    return a + b"}]

    def test_fenced_without_language_tag_stripped(self):
        """Plain ``` ... ``` wrapping is removed."""
        data = [{"prompt": "hi"}]
        source = data
        model = FakeModel(["```\nprint('hi')\n```"])
        result = translate.to_code(
            source,
            target_language="Python",
            fields=["prompt"],
            model=model,
        )
        assert result == [{"prompt": "print('hi')"}]

    def test_unfenced_response_left_alone(self):
        """A response without fences is returned (just stripped)."""
        data = [{"prompt": "hi"}]
        source = data
        model = FakeModel(["  print('hi')\n"])
        result = translate.to_code(
            source,
            target_language="Python",
            fields=["prompt"],
            model=model,
        )
        assert result == [{"prompt": "print('hi')"}]

    def test_inner_fences_preserved(self):
        """Fenced blocks nested inside the code are not touched."""
        data = [{"prompt": "print a markdown example"}]
        source = data
        inner = (
            "```python\n"
            "example = '''\n"
            "```\n"
            "code inside a string\n"
            "```\n"
            "'''\n"
            "print(example)\n"
            "```"
        )
        model = FakeModel([inner])
        result = translate.to_code(
            source,
            target_language="Python",
            fields=["prompt"],
            model=model,
        )
        expected = (
            "example = '''\n"
            "```\n"
            "code inside a string\n"
            "```\n"
            "'''\n"
            "print(example)"
        )
        assert result == [{"prompt": expected}]


class TestToCodeSourceLanguage:
    """source_language is woven into the to_code system prompt when given."""

    def test_source_language_included_when_given(self):
        """source_language is reflected in the system prompt."""
        data = [{"code": "def add(a, b): return a + b"}]
        source = data
        model = FakeModel(["fn add(a: i32, b: i32) -> i32 { a + b }"])
        # We don't care about the result here, and instead we just examine internally
        # how many times the model was called and what prompts were given.
        translate.to_code(
            source,
            target_language="Rust",
            fields=["code"],
            source_language="Python",
            model=model,
        )
        prompt = model.batch_calls[0][0]
        system = next(m for m in prompt if m["role"] == "system")
        assert "Python" in system["content"]
        assert "Rust" in system["content"]

    def test_source_language_absent_when_not_given(self):
        """Without source_language, no 'input is written in' phrase appears."""
        data = [{"prompt": "hello"}]
        source = data
        model = FakeModel(["print('hello')"])
        # We don't care about the result here, and instead we just examine internally
        # how many times the model was called and what prompts were given.
        translate.to_code(
            source,
            target_language="Python",
            fields=["prompt"],
            model=model,
        )
        prompt = model.batch_calls[0][0]
        system = next(m for m in prompt if m["role"] == "system")
        assert "input is written in" not in system["content"]
