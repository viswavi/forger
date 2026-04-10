"""Tests for pyforger.alter.translate.to_language."""

import pytest
from fixtures import make_source

from pyforger import run
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


class TestArgumentValidation:
    """to_language validates its arguments at call time."""

    def test_empty_fields_raises(self):
        """Empty fields list is rejected at call time."""
        source = make_source([{"instruction": "hi"}])
        model = FakeModel([])
        with pytest.raises(ValueError, match="fields"):
            translate.to_language(
                source,
                target_language="Spanish",
                fields=[],
                model=model,
            )

    def test_target_fields_length_mismatch_raises(self):
        """target_fields must be the same length as fields."""
        source = make_source([{"instruction": "hi", "response": "hello"}])
        model = FakeModel([])
        with pytest.raises(ValueError, match="target_fields"):
            translate.to_language(
                source,
                target_language="Spanish",
                fields=["instruction", "response"],
                target_fields=["instruction_es"],
                model=model,
            )

    def test_target_fields_equal_length_accepted(self):
        """Matched lengths should construct without error."""
        source = make_source([])
        model = FakeModel([])
        translate.to_language(
            source,
            target_language="Spanish",
            fields=["instruction", "response"],
            target_fields=["instruction_es", "response_es"],
            model=model,
        )


class TestSingleField:
    """Translating a single field, in place."""

    def test_overwrites_in_place(self):
        """With no target_fields, the source field is replaced."""
        data = [{"instruction": "hello"}]
        source = make_source(data)
        model = FakeModel(["hola"])
        node = translate.to_language(
            source,
            target_language="Spanish",
            fields=["instruction"],
            model=model,
        )
        result = run(node)
        assert result == [{"instruction": "hola"}]

    def test_multiple_items(self):
        """Each item in the input gets its own translated output."""
        data = [{"instruction": "hello"}, {"instruction": "goodbye"}]
        source = make_source(data)
        model = FakeModel(["hola", "adios"])
        node = translate.to_language(
            source,
            target_language="Spanish",
            fields=["instruction"],
            model=model,
        )
        result = run(node)
        assert result == [{"instruction": "hola"}, {"instruction": "adios"}]

    def test_other_keys_pass_through(self):
        """Fields not in the translate list are copied through unchanged."""
        data = [{"instruction": "hello", "id": 42, "tag": "greeting"}]
        source = make_source(data)
        model = FakeModel(["hola"])
        node = translate.to_language(
            source,
            target_language="Spanish",
            fields=["instruction"],
            model=model,
        )
        result = run(node)
        assert result == [{"instruction": "hola", "id": 42, "tag": "greeting"}]

    def test_leading_trailing_whitespace_stripped(self):
        """Model output is stripped before being written back."""
        data = [{"instruction": "hello"}]
        source = make_source(data)
        model = FakeModel(["  hola\n"])
        node = translate.to_language(
            source,
            target_language="Spanish",
            fields=["instruction"],
            model=model,
        )
        result = run(node)
        assert result == [{"instruction": "hola"}]

    def test_markdown_fences_in_content_preserved(self):
        """to_language must not strip code fences; only to_code does that."""
        data = [{"doc": "explain a function"}]
        source = make_source(data)
        translated = (
            "aqui esta la funcion:\n"
            "```python\n"
            "def foo():\n"
            "    return 1\n"
            "```"
        )
        model = FakeModel([translated])
        node = translate.to_language(
            source,
            target_language="Spanish",
            fields=["doc"],
            model=model,
        )
        result = run(node)
        assert result == [{"doc": translated}]


class TestMultiField:
    """Translating more than one field per item, one LLM call per field."""

    def test_both_fields_replaced(self):
        """Multi-field request translates every named field."""
        data = [{"instruction": "hello", "response": "hi there"}]
        source = make_source(data)
        # Two prompts for one item (instruction, then response).
        model = FakeModel(["hola", "hola, que tal"])
        node = translate.to_language(
            source,
            target_language="Spanish",
            fields=["instruction", "response"],
            model=model,
        )
        result = run(node)
        assert result == [{"instruction": "hola", "response": "hola, que tal"}]

    def test_one_call_per_field_per_item(self):
        """N items x M fields => N*M prompts in a single generate_batch call."""
        data = [
            {"instruction": "a", "response": "b"},
            {"instruction": "c", "response": "d"},
        ]
        source = make_source(data)
        model = FakeModel(["A", "B", "C", "D"])
        node = translate.to_language(
            source,
            target_language="French",
            fields=["instruction", "response"],
            model=model,
        )
        run(node)
        # One batch call, four prompts (2 items x 2 fields).
        assert len(model.batch_calls) == 1
        assert len(model.batch_calls[0]) == 4

    def test_prompt_order_is_item_major(self):
        """Prompts are laid out as item 0's fields, then item 1's, etc."""
        data = [
            {"instruction": "a", "response": "b"},
            {"instruction": "c", "response": "d"},
        ]
        source = make_source(data)
        model = FakeModel(["A", "B", "C", "D"])
        node = translate.to_language(
            source,
            target_language="French",
            fields=["instruction", "response"],
            model=model,
        )
        run(node)
        batch = model.batch_calls[0]
        user_msgs_per_prompt = [
            [m for m in prompt if m["role"] == "user"] for prompt in batch
        ]
        # Each prompt must carry exactly one user message, not a stack of them.
        assert all(len(msgs) == 1 for msgs in user_msgs_per_prompt)
        user_contents = [msgs[0]["content"] for msgs in user_msgs_per_prompt]
        assert user_contents == ["a", "b", "c", "d"]

    def test_results_written_back_in_order(self):
        """Translations land on the correct (item, field) slot."""
        data = [
            {"instruction": "a", "response": "b"},
            {"instruction": "c", "response": "d"},
        ]
        source = make_source(data)
        model = FakeModel(["A", "B", "C", "D"])
        node = translate.to_language(
            source,
            target_language="French",
            fields=["instruction", "response"],
            model=model,
        )
        result = run(node)
        assert result == [
            {"instruction": "A", "response": "B"},
            {"instruction": "C", "response": "D"},
        ]


class TestTargetFields:
    """target_fields preserves originals and writes to new keys."""

    def test_originals_preserved(self):
        """Source fields are kept and translations go to new keys."""
        data = [{"instruction": "hello", "response": "hi"}]
        source = make_source(data)
        model = FakeModel(["hola", "que tal"])
        node = translate.to_language(
            source,
            target_language="Spanish",
            fields=["instruction", "response"],
            target_fields=["instruction_es", "response_es"],
            model=model,
        )
        result = run(node)
        assert result == [
            {
                "instruction": "hello",
                "response": "hi",
                "instruction_es": "hola",
                "response_es": "que tal",
            }
        ]

    def test_different_target_name(self):
        """Single target_fields entry works the same as the multi-field case."""
        data = [{"instruction": "hello"}]
        source = make_source(data)
        model = FakeModel(["bonjour"])
        node = translate.to_language(
            source,
            target_language="French",
            fields=["instruction"],
            target_fields=["instruction_fr"],
            model=model,
        )
        result = run(node)
        assert result == [{"instruction": "hello", "instruction_fr": "bonjour"}]


class TestMissingSourceField:
    """Missing source fields raise KeyError via get_field, wrapped by the runner."""

    def test_missing_field_raises(self):
        """An item missing a listed field surfaces a KeyError via get_field."""
        data = [{"instruction": "hi"}]
        source = make_source(data)
        model = FakeModel([])
        node = translate.to_language(
            source,
            target_language="Spanish",
            fields=["instruction", "response"],
            model=model,
        )
        with pytest.raises(RuntimeError, match="missing required field 'response'"):
            run(node)


class TestModelErrors:
    """Model-side failure modes surface as loud errors with context."""

    def test_length_mismatch_too_few_raises(self):
        """Model returning fewer responses than prompts fails loudly."""
        data = [
            {"instruction": "hi"},
            {"instruction": "hello"},
        ]
        source = make_source(data)
        model = BrokenModel(["hola"])  # 1 response for 2 prompts
        node = translate.to_language(
            source,
            target_language="Spanish",
            fields=["instruction"],
            model=model,
        )
        with pytest.raises(RuntimeError, match="1 completions for 2 prompts"):
            run(node)

    def test_length_mismatch_too_many_raises(self):
        """Model returning more responses than prompts also fails loudly."""
        data = [{"instruction": "hi"}]
        source = make_source(data)
        model = BrokenModel(["hola", "extra"])  # 2 responses for 1 prompt
        node = translate.to_language(
            source,
            target_language="Spanish",
            fields=["instruction"],
            model=model,
        )
        with pytest.raises(RuntimeError, match="2 completions for 1 prompts"):
            run(node)

    def test_wrong_shape_raises(self):
        """Model returning flat list[str] instead of list[list[str]] fails."""

        class FlatModel:
            """Returns list[str] instead of list[list[str]]."""

            def generate_batch(self, prompts, *, n=1, **_):
                """Return a string per prompt, not a list of strings."""
                return ["hola" for _ in prompts]

        data = [{"instruction": "hi"}]
        source = make_source(data)
        node = translate.to_language(
            source,
            target_language="Spanish",
            fields=["instruction"],
            model=FlatModel(),
        )
        with pytest.raises(RuntimeError, match="expected list of completions"):
            run(node)

    def test_no_completions_for_prompt_raises(self):
        """A prompt that gets an empty choices list names item and field."""
        data = [{"instruction": "hi", "response": "hello"}]
        source = make_source(data)
        model = EmptyChoicesModel()
        node = translate.to_language(
            source,
            target_language="Spanish",
            fields=["instruction", "response"],
            model=model,
        )
        with pytest.raises(RuntimeError, match="item 0 field 'instruction'"):
            run(node)


class TestPromptContent:
    """System prompt and user message contain what they should."""

    def test_system_prompt_includes_target_language(self):
        """The target language string appears in the system message."""
        data = [{"instruction": "hello"}]
        source = make_source(data)
        model = FakeModel(["bonjour"])
        node = translate.to_language(
            source,
            target_language="French",
            fields=["instruction"],
            model=model,
        )
        run(node)
        prompt = model.batch_calls[0][0]
        assert isinstance(prompt, list)
        system = next(m for m in prompt if m["role"] == "system")
        assert "French" in system["content"]

    def test_user_message_is_raw_field_value(self):
        """The user message is the source field value, not a wrapped version."""
        data = [{"instruction": "what is 2+2?"}]
        source = make_source(data)
        model = FakeModel(["cuanto es 2+2?"])
        node = translate.to_language(
            source,
            target_language="Spanish",
            fields=["instruction"],
            model=model,
        )
        run(node)
        prompt = model.batch_calls[0][0]
        user = next(m for m in prompt if m["role"] == "user")
        assert user["content"] == "what is 2+2?"

    def test_each_prompt_has_one_field_value(self):
        """Multi-field translation sends each field as its own prompt."""
        data = [{"instruction": "A", "response": "B"}]
        source = make_source(data)
        model = FakeModel(["a", "b"])
        node = translate.to_language(
            source,
            target_language="Spanish",
            fields=["instruction", "response"],
            model=model,
        )
        run(node)
        batch = model.batch_calls[0]
        assert len(batch) == 2
        user_msgs_per_prompt = [[m for m in p if m["role"] == "user"] for p in batch]
        # Each prompt carries exactly one user message with one field value.
        assert all(len(msgs) == 1 for msgs in user_msgs_per_prompt)
        user_contents = [msgs[0]["content"] for msgs in user_msgs_per_prompt]
        assert user_contents == ["A", "B"]


class TestNodeMetadata:
    """The returned DataSpec has the expected shape."""

    def test_node_type_is_alter(self):
        """The constructed node is an ALTER node."""
        from pyforger.core import NodeType

        source = make_source([])
        model = FakeModel([])
        node = translate.to_language(
            source,
            target_language="Spanish",
            fields=["instruction"],
            model=model,
        )
        assert node.node_type == NodeType.ALTER

    def test_node_name(self):
        """The constructed node has the documented debug name."""
        source = make_source([])
        model = FakeModel([])
        node = translate.to_language(
            source,
            target_language="Spanish",
            fields=["instruction"],
            model=model,
        )
        assert node.name == "alter_translate_to_language"


class TestEdgeCases:
    """Boundary conditions."""

    def test_empty_input_skips_llm_call(self):
        """No data means no generate_batch call is made."""
        source = make_source([])
        model = FakeModel([])
        node = translate.to_language(
            source,
            target_language="Spanish",
            fields=["instruction"],
            model=model,
        )
        result = run(node)
        assert result == []
        assert model.batch_calls == []

    def test_empty_string_field_value_passed_through(self):
        """An empty string is still sent to the model and its output used."""
        data = [{"instruction": ""}]
        source = make_source(data)
        model = FakeModel([""])
        node = translate.to_language(
            source,
            target_language="Spanish",
            fields=["instruction"],
            model=model,
        )
        result = run(node)
        assert result == [{"instruction": ""}]

    def test_multiline_field_content(self):
        """Multi-line text is translated as one piece, no extra processing."""
        data = [{"instruction": "line one\nline two\nline three"}]
        source = make_source(data)
        model = FakeModel(["linea uno\nlinea dos\nlinea tres"])
        node = translate.to_language(
            source,
            target_language="Spanish",
            fields=["instruction"],
            model=model,
        )
        result = run(node)
        assert result == [{"instruction": "linea uno\nlinea dos\nlinea tres"}]

    def test_input_dicts_not_mutated(self):
        """The alter step returns new dicts and a new list; source is untouched."""
        original = {"instruction": "hello", "id": 1}
        data = [original]
        source = make_source(data)
        model = FakeModel(["hola"])
        node = translate.to_language(
            source,
            target_language="Spanish",
            fields=["instruction"],
            model=model,
        )
        result = run(node)
        assert original == {"instruction": "hello", "id": 1}
        assert result[0] is not original
        assert result is not data


class TestToCodeArgumentValidation:
    """to_code validates its arguments at call time."""

    def test_empty_fields_raises(self):
        """Empty fields list is rejected at call time."""
        source = make_source([{"prompt": "add two numbers"}])
        model = FakeModel([])
        with pytest.raises(ValueError, match="translate.to_code.*fields"):
            translate.to_code(
                source,
                target_language="Python",
                fields=[],
                model=model,
            )

    def test_target_fields_length_mismatch_raises(self):
        """target_fields must be the same length as fields."""
        source = make_source([{"prompt": "x", "spec": "y"}])
        model = FakeModel([])
        with pytest.raises(ValueError, match="translate.to_code.*target_fields"):
            translate.to_code(
                source,
                target_language="Python",
                fields=["prompt", "spec"],
                target_fields=["code"],
                model=model,
            )


class TestToCodeSingleField:
    """Translating a single description field into code."""

    def test_overwrites_in_place(self):
        """With no target_fields, the source field is replaced with code."""
        data = [{"prompt": "add two numbers"}]
        source = make_source(data)
        model = FakeModel(["def add(a, b):\n    return a + b"])
        node = translate.to_code(
            source,
            target_language="Python",
            fields=["prompt"],
            model=model,
        )
        result = run(node)
        assert result == [{"prompt": "def add(a, b):\n    return a + b"}]

    def test_target_fields_preserves_original(self):
        """Original prompt is kept; generated code is written to target."""
        data = [{"prompt": "reverse a string"}]
        source = make_source(data)
        model = FakeModel(["def reverse(s):\n    return s[::-1]"])
        node = translate.to_code(
            source,
            target_language="Python",
            fields=["prompt"],
            target_fields=["code"],
            model=model,
        )
        result = run(node)
        assert result == [
            {
                "prompt": "reverse a string",
                "code": "def reverse(s):\n    return s[::-1]",
            }
        ]

    def test_other_keys_pass_through(self):
        """Fields not in the translate list are copied through unchanged."""
        data = [{"prompt": "hello world", "id": 7, "difficulty": "easy"}]
        source = make_source(data)
        model = FakeModel(["print('hello world')"])
        node = translate.to_code(
            source,
            target_language="Python",
            fields=["prompt"],
            target_fields=["code"],
            model=model,
        )
        result = run(node)
        assert result == [
            {
                "prompt": "hello world",
                "id": 7,
                "difficulty": "easy",
                "code": "print('hello world')",
            }
        ]


class TestToCodeMultiField:
    """Translating more than one field per item into code."""

    def test_both_fields_replaced(self):
        """Every named field gets its own generated code."""
        data = [{"setup": "read input", "body": "sum the numbers"}]
        source = make_source(data)
        model = FakeModel(
            [
                "nums = [int(x) for x in input().split()]",
                "print(sum(nums))",
            ]
        )
        node = translate.to_code(
            source,
            target_language="Python",
            fields=["setup", "body"],
            model=model,
        )
        result = run(node)
        assert result == [
            {
                "setup": "nums = [int(x) for x in input().split()]",
                "body": "print(sum(nums))",
            }
        ]

    def test_one_batch_call_for_all_items_and_fields(self):
        """N items x M fields => one generate_batch with N*M prompts."""
        data = [
            {"setup": "a", "body": "b"},
            {"setup": "c", "body": "d"},
        ]
        source = make_source(data)
        model = FakeModel(["A", "B", "C", "D"])
        node = translate.to_code(
            source,
            target_language="Rust",
            fields=["setup", "body"],
            model=model,
        )
        run(node)
        assert len(model.batch_calls) == 1
        assert len(model.batch_calls[0]) == 4


class TestToCodePromptContent:
    """System prompt reflects target_language and optional source_language."""

    def test_target_language_in_system_prompt(self):
        """The target programming language appears in the system message."""
        data = [{"prompt": "hello"}]
        source = make_source(data)
        model = FakeModel(["console.log('hello')"])
        node = translate.to_code(
            source,
            target_language="TypeScript",
            fields=["prompt"],
            model=model,
        )
        run(node)
        prompt = model.batch_calls[0][0]
        system = next(m for m in prompt if m["role"] == "system")
        assert "TypeScript" in system["content"]

    def test_source_language_included_when_given(self):
        """source_language is woven into the system prompt when provided."""
        data = [{"code": "def add(a, b): return a + b"}]
        source = make_source(data)
        model = FakeModel(["fn add(a: i32, b: i32) -> i32 { a + b }"])
        node = translate.to_code(
            source,
            target_language="Rust",
            fields=["code"],
            source_language="Python",
            model=model,
        )
        run(node)
        prompt = model.batch_calls[0][0]
        system = next(m for m in prompt if m["role"] == "system")
        assert "Python" in system["content"]
        assert "Rust" in system["content"]

    def test_source_language_absent_when_not_given(self):
        """Without source_language, no 'input is written in' phrase appears."""
        data = [{"prompt": "hello"}]
        source = make_source(data)
        model = FakeModel(["print('hello')"])
        node = translate.to_code(
            source,
            target_language="Python",
            fields=["prompt"],
            model=model,
        )
        run(node)
        prompt = model.batch_calls[0][0]
        system = next(m for m in prompt if m["role"] == "system")
        assert "input is written in" not in system["content"]

    def test_user_message_is_raw_field_value(self):
        """The user message is the raw source field value, not wrapped."""
        data = [{"prompt": "compute the factorial of n"}]
        source = make_source(data)
        model = FakeModel(["def fact(n): return 1 if n <= 1 else n * fact(n-1)"])
        node = translate.to_code(
            source,
            target_language="Python",
            fields=["prompt"],
            model=model,
        )
        run(node)
        prompt = model.batch_calls[0][0]
        user = next(m for m in prompt if m["role"] == "user")
        assert user["content"] == "compute the factorial of n"


class TestToCodeFenceStripping:
    """to_code strips outer markdown code fences the model adds unprompted."""

    def test_fenced_with_language_tag_stripped(self):
        """```python ... ``` wrapping is removed."""
        data = [{"prompt": "add"}]
        source = make_source(data)
        model = FakeModel(["```python\ndef add(a, b):\n    return a + b\n```"])
        node = translate.to_code(
            source,
            target_language="Python",
            fields=["prompt"],
            model=model,
        )
        result = run(node)
        assert result == [{"prompt": "def add(a, b):\n    return a + b"}]

    def test_fenced_without_language_tag_stripped(self):
        """Plain ``` ... ``` wrapping is removed."""
        data = [{"prompt": "hi"}]
        source = make_source(data)
        model = FakeModel(["```\nprint('hi')\n```"])
        node = translate.to_code(
            source,
            target_language="Python",
            fields=["prompt"],
            model=model,
        )
        result = run(node)
        assert result == [{"prompt": "print('hi')"}]

    def test_unfenced_response_left_alone(self):
        """A response without fences is returned (just stripped)."""
        data = [{"prompt": "hi"}]
        source = make_source(data)
        model = FakeModel(["  print('hi')\n"])
        node = translate.to_code(
            source,
            target_language="Python",
            fields=["prompt"],
            model=model,
        )
        result = run(node)
        assert result == [{"prompt": "print('hi')"}]

    def test_inner_fences_preserved(self):
        """Fenced blocks nested inside the code are not touched."""
        data = [{"prompt": "print a markdown example"}]
        source = make_source(data)
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
        node = translate.to_code(
            source,
            target_language="Python",
            fields=["prompt"],
            model=model,
        )
        result = run(node)
        expected = (
            "example = '''\n"
            "```\n"
            "code inside a string\n"
            "```\n"
            "'''\n"
            "print(example)"
        )
        assert result == [{"prompt": expected}]


class TestToCodeMissingSourceField:
    """Missing source fields raise KeyError via get_field, wrapped by the runner."""

    def test_missing_field_raises(self):
        """An item missing a listed field surfaces through get_field."""
        data = [{"prompt": "hi"}]
        source = make_source(data)
        model = FakeModel([])
        node = translate.to_code(
            source,
            target_language="Python",
            fields=["prompt", "spec"],
            model=model,
        )
        with pytest.raises(RuntimeError, match="missing required field 'spec'"):
            run(node)


class TestToCodeNodeMetadata:
    """The returned DataSpec has the expected shape for to_code."""

    def test_node_type_is_alter(self):
        """The constructed node is an ALTER node."""
        from pyforger.core import NodeType

        source = make_source([])
        model = FakeModel([])
        node = translate.to_code(
            source,
            target_language="Python",
            fields=["prompt"],
            model=model,
        )
        assert node.node_type == NodeType.ALTER

    def test_node_name(self):
        """The constructed node has the documented debug name."""
        source = make_source([])
        model = FakeModel([])
        node = translate.to_code(
            source,
            target_language="Python",
            fields=["prompt"],
            model=model,
        )
        assert node.name == "alter_translate_to_code"
