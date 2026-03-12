"""Tests for syntax validation filter operations."""

import pytest
from fixtures import make_source

from pyforger import DataSpec, run
from pyforger.filter import by_syntax


class TestBySyntaxBasic:
    """by_syntax general behavior tests."""

    def test_returns_dataspec(self):
        """by_syntax returns a DataSpec node."""
        source = make_source([])
        assert isinstance(by_syntax(source), DataSpec)

    def test_default_field_is_output(self):
        """Default field is 'output'."""
        data = [{"output": '{"a": 1}'}]
        source = make_source(data)
        result = run(by_syntax(source))
        assert result == data

    def test_custom_field(self):
        """Custom field name is used for validation."""
        data = [{"response": '{"a": 1}'}]
        source = make_source(data)
        result = run(by_syntax(source, field="response"))
        assert result == data

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = make_source([])
        result = run(by_syntax(source))
        assert result == []

    def test_none_field_value_removed(self):
        """Item with None in the field is filtered out."""
        data = [{"output": None}]
        source = make_source(data)
        result = run(by_syntax(source))
        assert result == []

    def test_empty_string_removed(self):
        """Empty string is rejected for all languages."""
        data = [{"output": ""}]
        source = make_source(data)
        for lang in ("json", "yaml", "python", "xml"):
            result = run(by_syntax(source, lang=lang))
            assert result == [], f"empty string should be rejected for {lang}"

    def test_whitespace_only_removed(self):
        """Whitespace-only string is rejected for all languages."""
        data = [{"output": "   \n\t  "}]
        source = make_source(data)
        for lang in ("json", "yaml", "python", "xml"):
            result = run(by_syntax(source, lang=lang))
            assert result == [], f"whitespace should be rejected for {lang}"

    def test_missing_field_raises_error(self):
        """Missing expected field raises an error."""
        data = [{"instruction": "hello"}]
        source = make_source(data)
        with pytest.raises(RuntimeError, match="output"):
            run(by_syntax(source))

    def test_unsupported_lang_raises_value_error(self):
        """Unsupported lang raises ValueError."""
        source = make_source([])
        with pytest.raises(ValueError, match="rust"):
            by_syntax(source, lang="rust")

    def test_required_keys_with_non_json_raises(self):
        """required_keys with non-JSON lang raises ValueError."""
        source = make_source([])
        with pytest.raises(ValueError, match="required_keys"):
            by_syntax(source, lang="yaml", required_keys=["a"])

    def test_json_schema_with_non_json_raises(self):
        """json_schema with non-JSON lang raises ValueError."""
        source = make_source([])
        with pytest.raises(ValueError, match="json_schema"):
            by_syntax(source, lang="yaml", json_schema={})


class TestBySyntaxJson:
    """by_syntax with lang='json' validates JSON syntax."""

    def test_valid_json_kept(self):
        """Valid JSON string passes the filter."""
        data = [{"output": '{"name": "Alice", "age": 30}'}]
        source = make_source(data)
        result = run(by_syntax(source, lang="json"))
        assert result == data

    def test_invalid_json_removed(self):
        """Malformed JSON string is filtered out."""
        data = [
            {"output": '{"valid": true}'},
            {"output": "{not json at all"},
        ]
        source = make_source(data)
        result = run(by_syntax(source, lang="json"))
        assert result == [data[0]]

    def test_json_array_valid(self):
        """JSON array is valid."""
        data = [{"output": "[1, 2, 3]"}]
        source = make_source(data)
        result = run(by_syntax(source, lang="json"))
        assert result == data

    def test_json_scalar_valid(self):
        """JSON scalar string is valid."""
        data = [{"output": '"hello"'}]
        source = make_source(data)
        result = run(by_syntax(source, lang="json"))
        assert result == data

    def test_required_keys_pass(self):
        """JSON with all required keys passes."""
        data = [{"output": '{"name": "Alice", "age": 30}'}]
        source = make_source(data)
        result = run(by_syntax(source, lang="json", required_keys=["name"]))
        assert result == data

    def test_required_keys_missing_key(self):
        """JSON missing a required key is filtered out."""
        data = [{"output": '{"age": 30}'}]
        source = make_source(data)
        result = run(by_syntax(source, lang="json", required_keys=["name"]))
        assert result == []

    def test_required_keys_non_object(self):
        """JSON non-object with required_keys is filtered out."""
        data = [{"output": "[1, 2, 3]"}]
        source = make_source(data)
        result = run(by_syntax(source, lang="json", required_keys=["name"]))
        assert result == []

    def test_json_schema_valid(self):
        """JSON conforming to schema passes."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        data = [{"output": '{"name": "Alice"}'}]
        source = make_source(data)
        result = run(by_syntax(source, lang="json", json_schema=schema))
        assert result == data

    def test_json_schema_invalid(self):
        """JSON violating schema is filtered out."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        data = [{"output": '{"age": 30}'}]
        source = make_source(data)
        result = run(by_syntax(source, lang="json", json_schema=schema))
        assert result == []

    def test_required_keys_and_schema_combined(self):
        """Both required_keys and json_schema are applied."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        data = [
            {"output": '{"name": "Alice", "role": "admin"}'},
            {"output": '{"name": "Bob"}'},
            {"output": '{"role": "user"}'},
        ]
        source = make_source(data)
        result = run(
            by_syntax(
                source,
                lang="json",
                required_keys=["role"],
                json_schema=schema,
            )
        )
        assert result == [data[0]]


class TestBySyntaxYaml:
    """by_syntax with lang='yaml' validates YAML syntax."""

    def test_valid_yaml_kept(self):
        """Valid YAML passes the filter."""
        data = [{"output": "key: value\nlist:\n  - item1\n"}]
        source = make_source(data)
        result = run(by_syntax(source, lang="yaml"))
        assert result == data

    def test_invalid_yaml_removed(self):
        """Malformed YAML is filtered out."""
        data = [{"output": "key: [unclosed"}]
        source = make_source(data)
        result = run(by_syntax(source, lang="yaml"))
        assert result == []

    def test_yaml_plain_string_valid(self):
        """Plain text is valid YAML (parsed as a scalar)."""
        data = [{"output": "just some plain text"}]
        source = make_source(data)
        result = run(by_syntax(source, lang="yaml"))
        assert result == data


class TestBySyntaxPython:
    """by_syntax with lang='python' validates Python syntax."""

    def test_valid_python_kept(self):
        """Valid Python code passes the filter."""
        data = [{"output": "x = 1\nprint(x)"}]
        source = make_source(data)
        result = run(by_syntax(source, lang="python"))
        assert result == data

    def test_invalid_python_removed(self):
        """Syntactically invalid Python is filtered out."""
        data = [{"output": "def f(\n"}]
        source = make_source(data)
        result = run(by_syntax(source, lang="python"))
        assert result == []

    def test_expression_valid(self):
        """A bare expression is valid Python."""
        data = [{"output": "2 + 2"}]
        source = make_source(data)
        result = run(by_syntax(source, lang="python"))
        assert result == data


class TestBySyntaxXml:
    """by_syntax with lang='xml' validates XML syntax."""

    def test_valid_xml_kept(self):
        """Valid XML passes the filter."""
        data = [{"output": "<root><child/></root>"}]
        source = make_source(data)
        result = run(by_syntax(source, lang="xml"))
        assert result == data

    def test_invalid_xml_removed(self):
        """Malformed XML is filtered out."""
        data = [{"output": "<root><unclosed>"}]
        source = make_source(data)
        result = run(by_syntax(source, lang="xml"))
        assert result == []

    def test_xml_with_attributes(self):
        """XML with attributes is valid."""
        data = [{"output": '<root attr="val"/>'}]
        source = make_source(data)
        result = run(by_syntax(source, lang="xml"))
        assert result == data


class TestBySyntaxMixedData:
    """Integration tests with mixed valid and invalid data."""

    def test_filters_mixed_valid_and_invalid(self):
        """Only valid items are kept from a mixed list."""
        data = [
            {"output": '{"a": 1}'},
            {"output": "not json"},
            {"output": '{"b": 2}'},
            {"output": "{broken"},
        ]
        source = make_source(data)
        result = run(by_syntax(source, lang="json"))
        assert result == [data[0], data[2]]

    def test_all_invalid_returns_empty(self):
        """All invalid items results in empty list."""
        data = [
            {"output": "{bad"},
            {"output": "also{bad"},
        ]
        source = make_source(data)
        result = run(by_syntax(source, lang="json"))
        assert result == []
