"""Syntax validation filter operations for Forger.

Filters that remove items whose text field is not syntactically valid
in a specified format (JSON, YAML, Python, XML).

Note: YAML is extremely permissive — most plain text parses as a valid
YAML scalar. The YAML validator is mainly useful for catching broken
structured YAML (e.g. unclosed brackets), not for rejecting free-form text.
"""
from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import Any

from pyforger.core import DataSpec, NodeType

_SUPPORTED_LANGS = frozenset({"json", "yaml", "python", "xml"})


def by_syntax(
    source: DataSpec,
    *,
    lang: str = "json",
    field: str = "output",
    required_keys: list[str] | None = None,
    json_schema: dict[str, Any] | None = None,
) -> DataSpec:
    """Remove items whose field value is not valid syntax in the given language.

    Supports JSON, YAML, Python, and XML. Empty or whitespace-only
    values are always rejected regardless of language.
    For JSON, optional deeper validation via required_keys or json_schema.

    Note: YAML is very permissive — most plain text is valid YAML.

    Args:
        source: Upstream DataSpec providing list[dict] data.
        lang: Language/format to validate against.
            One of: json, yaml,python, xml.
        field: Key in each dict containing the text to validate.
        required_keys: JSON-only. If provided, parsed JSON must be a
            dict containing all listed keys.
        json_schema: JSON-only. If provided, parsed JSON must conform
            to this JSON Schema.

    Returns:
        A FILTER DataSpec that keeps only syntactically valid items.

    Raises:
        ValueError: If lang is not supported, or if required_keys /
            json_schema is used with a non-JSON lang.
        ImportError: If lang='yaml' and PyYAML is not installed, or
            if json_schema is used and jsonschema is not installed.
    """
    if lang not in _SUPPORTED_LANGS:
        raise ValueError(
            f"Unsupported lang {lang!r}. " f"Supported: {sorted(_SUPPORTED_LANGS)}"
        )
    if required_keys is not None and lang != "json":
        raise ValueError("required_keys is only supported when lang='json'")
    if json_schema is not None and lang != "json":
        raise ValueError("json_schema is only supported when lang='json'")

    # Import optional dependencies eagerly so users get errors at
    # construction time, not when the pipeline runs.
    if lang == "yaml":
        try:
            import yaml  # type: ignore[import]  # noqa: F401
        except ImportError:
            raise ImportError(
                "PyYAML is required for lang='yaml'. "
                "Install it with: pip install pyyaml"
            ) from None
    if json_schema is not None:
        try:
            import jsonschema  # noqa: F401
        except ImportError:
            raise ImportError(
                "jsonschema is required for json_schema validation. "
                "Install it with: pip install jsonschema"
            ) from None

    def _execute(data: list[dict]) -> list[dict]:
        return [
            item
            for item in data
            if _validate(item[field], lang, required_keys, json_schema)
        ]

    return DataSpec(NodeType.FILTER, source, _execute, "filter_by_syntax")


def _validate(
    text: str | None,
    lang: str,
    required_keys: list[str] | None,
    json_schema: dict[str, Any] | None,
) -> bool:
    """Route validation to the appropriate language validator."""
    if text is None or not text.strip():
        return False
    # JSON is handled separately because it accepts extra validation
    # args (required_keys, json_schema). Other langs use _VALIDATORS.
    if lang == "json":
        return _validate_json(text, required_keys, json_schema)
    return _VALIDATORS[lang](text)


def _validate_json(
    text: str,
    required_keys: list[str] | None,
    json_schema: dict[str, Any] | None,
) -> bool:
    """Validate JSON syntax, optionally checking keys or schema."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False

    if required_keys is not None:
        if not isinstance(parsed, dict):
            return False
        if not all(k in parsed for k in required_keys):
            return False

    if json_schema is not None:
        import jsonschema

        try:
            jsonschema.validate(parsed, json_schema)
        except jsonschema.ValidationError:
            return False

    return True


def _validate_yaml(text: str) -> bool:
    """Validate YAML syntax."""
    import yaml  # type: ignore[import]

    try:
        yaml.safe_load(text)
    except yaml.YAMLError:
        return False
    return True


def _validate_python(text: str) -> bool:
    """Validate Python syntax."""
    try:
        ast.parse(text)
    except SyntaxError:
        return False
    return True


def _validate_xml(text: str) -> bool:
    """Validate XML syntax."""
    try:
        ET.fromstring(text)
    except ET.ParseError:
        return False
    return True


_VALIDATORS: dict[str, Callable[[str], bool]] = {
    "yaml": _validate_yaml,
    "python": _validate_python,
    "xml": _validate_xml,
}
