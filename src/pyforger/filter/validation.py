"""Syntax validation filter.

Drops items whose text field is not syntactically valid in the chosen
language (json, yaml, python, xml).

YAML is permissive — most plain text is valid YAML. Useful mainly for
catching broken structured YAML.
"""
from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import Any

from pyforger.op import op
from pyforger.utils import get_field

_SUPPORTED_LANGS = frozenset({"json", "yaml", "python", "xml"})


@op(name="filter.by_syntax")
def by_syntax(
    items: list[dict[str, Any]],
    *,
    lang: str = "json",
    field: str,
    required_keys: list[str] | None = None,
    json_schema: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Keep items whose ``field`` parses as valid ``lang``."""
    assert (
        lang in _SUPPORTED_LANGS
    ), f"Unsupported lang {lang!r}. Supported: {sorted(_SUPPORTED_LANGS)}"
    assert (
        required_keys is None or lang == "json"
    ), "required_keys is only supported when lang='json'"
    assert (
        json_schema is None or lang == "json"
    ), "json_schema is only supported when lang='json'"

    if lang == "yaml":
        try:
            import yaml  # noqa: F401
        except ImportError:
            raise ImportError(
                "PyYAML is required for lang='yaml'. Install: pip install pyyaml"
            ) from None
    if json_schema is not None:
        try:
            import jsonschema  # noqa: F401
        except ImportError:
            raise ImportError(
                "jsonschema is required for json_schema validation. "
                "Install: pip install jsonschema"
            ) from None

    return [
        item
        for item in items
        if _validate(
            get_field(item, field, "by_syntax"), lang, required_keys, json_schema
        )
    ]


def _validate(
    text: str | None,
    lang: str,
    required_keys: list[str] | None,
    json_schema: dict[str, Any] | None,
) -> bool:
    if text is None or not text.strip():
        return False
    if lang == "json":
        return _validate_json(text, required_keys, json_schema)
    return _VALIDATORS[lang](text)


def _validate_json(
    text: str,
    required_keys: list[str] | None,
    json_schema: dict[str, Any] | None,
) -> bool:
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
    import yaml

    try:
        yaml.safe_load(text)
    except yaml.YAMLError:
        return False
    return True


def _validate_python(text: str) -> bool:
    try:
        ast.parse(text)
    except SyntaxError:
        return False
    return True


def _validate_xml(text: str) -> bool:
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
