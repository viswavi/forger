"""Tests for syntax validation filter operations."""

import pytest

from pyforger.filter import by_syntax


class TestBySyntaxBasic:
    """by_syntax general behavior tests."""

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = []
        result = by_syntax(source, field="response")
        assert result == []

    def test_none_field_value_removed(self):
        """Item with None in the field is filtered out."""
        data = [{"response": None}]
        source = data
        result = by_syntax(source, field="response")
        assert result == []

    def test_empty_string_removed(self):
        """Empty string is rejected for all languages."""
        data = [{"response": ""}]
        source = data
        for lang in ("json", "yaml", "python", "xml"):
            result = by_syntax(source, lang=lang, field="response")
            assert result == [], f"empty string should be rejected for {lang}"

    def test_whitespace_only_removed(self):
        """Whitespace-only string is rejected for all languages."""
        data = [{"response": "   \n\t  "}]
        source = data
        for lang in ("json", "yaml", "python", "xml"):
            result = by_syntax(source, lang=lang, field="response")
            assert result == [], f"whitespace should be rejected for {lang}"

    def test_missing_field_raises_error(self):
        """Missing expected field raises an error."""
        data = [{"instruction": "hello"}]
        source = data
        with pytest.raises(KeyError, match="missing required field 'response'"):
            by_syntax(source, field="response")

    def test_unsupported_lang_raises_value_error(self):
        """Unsupported lang raises AssertionError."""
        source = []
        with pytest.raises(AssertionError, match="rust"):
            by_syntax(source, lang="rust", field="response")

    def test_required_keys_with_non_json_raises(self):
        """required_keys with non-JSON lang raises AssertionError."""
        source = []
        with pytest.raises(AssertionError, match="required_keys"):
            by_syntax(source, lang="yaml", field="response", required_keys=["a"])

    def test_json_schema_with_non_json_raises(self):
        """json_schema with non-JSON lang raises AssertionError."""
        source = []
        with pytest.raises(AssertionError, match="json_schema"):
            by_syntax(source, lang="yaml", field="response", json_schema={})


class TestBySyntaxJson:
    """by_syntax with lang='json' validates JSON syntax."""

    def test_valid_json_kept(self):
        """Valid JSON string passes the filter."""
        data = [{"response": '{"name": "Alice", "age": 30}'}]
        source = data
        result = by_syntax(source, lang="json", field="response")
        assert result == data

    def test_invalid_json_removed(self):
        """Malformed JSON string is filtered out."""
        data = [
            {"response": '{"valid": true}'},
            {"response": "{not json at all"},
        ]
        source = data
        result = by_syntax(source, lang="json", field="response")
        assert result == [data[0]]

    def test_json_array_valid(self):
        """JSON array is valid."""
        data = [{"response": "[1, 2, 3]"}]
        source = data
        result = by_syntax(source, lang="json", field="response")
        assert result == data

    def test_json_scalar_valid(self):
        """JSON scalar string is valid."""
        data = [{"response": '"hello"'}]
        source = data
        result = by_syntax(source, lang="json", field="response")
        assert result == data

    def test_required_keys_pass(self):
        """JSON with all required keys passes."""
        data = [{"response": '{"name": "Alice", "age": 30}'}]
        source = data
        result = by_syntax(
            source, lang="json", field="response", required_keys=["name"]
        )
        assert result == data

    def test_required_keys_missing_key(self):
        """JSON missing a required key is filtered out."""
        data = [{"response": '{"age": 30}'}]
        source = data
        result = by_syntax(
            source, lang="json", field="response", required_keys=["name"]
        )
        assert result == []

    def test_required_keys_non_object(self):
        """JSON non-object with required_keys is filtered out."""
        data = [{"response": "[1, 2, 3]"}]
        source = data
        result = by_syntax(
            source, lang="json", field="response", required_keys=["name"]
        )
        assert result == []

    def test_json_schema_valid(self):
        """JSON conforming to schema passes."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        data = [{"response": '{"name": "Alice"}'}]
        source = data
        result = by_syntax(source, lang="json", field="response", json_schema=schema)
        assert result == data

    def test_json_schema_invalid(self):
        """JSON violating schema is filtered out."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        data = [{"response": '{"age": 30}'}]
        source = data
        result = by_syntax(source, lang="json", field="response", json_schema=schema)
        assert result == []

    def test_required_keys_and_schema_combined(self):
        """Both required_keys and json_schema are applied."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        data = [
            {"response": '{"name": "Alice", "role": "admin"}'},
            {"response": '{"name": "Bob"}'},
            {"response": '{"role": "user"}'},
        ]
        source = data
        result = by_syntax(
            source,
            lang="json",
            field="response",
            required_keys=["role"],
            json_schema=schema,
        )
        assert result == [data[0]]


class TestBySyntaxYaml:
    """by_syntax with lang='yaml' validates YAML syntax."""

    def test_valid_yaml_kept(self):
        """Valid YAML passes the filter."""
        data = [{"response": "key: value\nlist:\n  - item1\n"}]
        source = data
        result = by_syntax(source, lang="yaml", field="response")
        assert result == data

    def test_invalid_yaml_removed(self):
        """Malformed YAML is filtered out."""
        data = [{"response": "key: [unclosed"}]
        source = data
        result = by_syntax(source, lang="yaml", field="response")
        assert result == []

    def test_yaml_plain_string_valid(self):
        """Plain text is valid YAML (parsed as a scalar)."""
        data = [{"response": "just some plain text"}]
        source = data
        result = by_syntax(source, lang="yaml", field="response")
        assert result == data


class TestBySyntaxPython:
    """by_syntax with lang='python' validates Python syntax."""

    def test_valid_python_kept(self):
        """Valid Python code passes the filter."""
        data = [{"response": "x = 1\nprint(x)"}]
        source = data
        result = by_syntax(source, lang="python", field="response")
        assert result == data

    def test_invalid_python_removed(self):
        """Syntactically invalid Python is filtered out."""
        data = [{"response": "def f(\n"}]
        source = data
        result = by_syntax(source, lang="python", field="response")
        assert result == []

    def test_expression_valid(self):
        """A bare expression is valid Python."""
        data = [{"response": "2 + 2"}]
        source = data
        result = by_syntax(source, lang="python", field="response")
        assert result == data


class TestBySyntaxXml:
    """by_syntax with lang='xml' validates XML syntax."""

    def test_valid_xml_kept(self):
        """Valid XML passes the filter."""
        data = [{"response": "<root><child/></root>"}]
        source = data
        result = by_syntax(source, lang="xml", field="response")
        assert result == data

    def test_invalid_xml_removed(self):
        """Malformed XML is filtered out."""
        data = [{"response": "<root><unclosed>"}]
        source = data
        result = by_syntax(source, lang="xml", field="response")
        assert result == []

    def test_xml_with_attributes(self):
        """XML with attributes is valid."""
        data = [{"response": '<root attr="val"/>'}]
        source = data
        result = by_syntax(source, lang="xml", field="response")
        assert result == data


class TestBySyntaxMixedData:
    """Integration tests with mixed valid and invalid data."""

    def test_filters_mixed_valid_and_invalid(self):
        """Only valid items are kept from a mixed list."""
        data = [
            {"response": '{"a": 1}'},
            {"response": "not json"},
            {"response": '{"b": 2}'},
            {"response": "{broken"},
        ]
        source = data
        result = by_syntax(source, lang="json", field="response")
        assert result == [data[0], data[2]]

    def test_all_invalid_returns_empty(self):
        """All invalid items results in empty list."""
        data = [
            {"response": "{bad"},
            {"response": "also{bad"},
        ]
        source = data
        result = by_syntax(source, lang="json", field="response")
        assert result == []
