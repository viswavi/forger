"""Translation alter operations for Forger.

to_language translates text fields on each item of a list[dict] into a
target natural language. to_code converts natural language descriptions
(or source code in another programming language) into code in a target
programming language. Both use a Model for generation.
"""
from __future__ import annotations

from collections.abc import Callable

from pyforger.acquire.model import Message, Model, Prompt
from pyforger.utils import get_field

SYSTEM_PROMPT_LANGUAGE = (
    "You are a professional translator. Translate the user's text into "
    "{target_language}. Output only the translated text with no preamble, "
    "commentary, or quotation marks. Preserve formatting such as code "
    "blocks, markdown, placeholders, and URLs unchanged."
)

SYSTEM_PROMPT_CODE = (
    "You are an expert programmer.{source_hint} Convert the user's text "
    "into working {target_language} code. Output only the code itself, "
    "with no explanation, no markdown code fences, and no surrounding prose."
)


def to_language(
    source: list[dict],
    *,
    target_language: str,
    fields: list[str],
    model: Model,
    target_fields: list[str] | None = None,
) -> list[dict]:
    """Translate text fields on each item into target_language using a Model.

    Each (item, field) pair is one LLM call, all dispatched through a single
    generate_batch so they run in parallel.

    Args:
        source: Upstream list of dictioanries
        target_language: Language name to translate into, passed through to
            the model prompt (e.g. "Spanish", "Brazilian Portuguese", "zh-Hans").
        fields: Keys whose values should be translated.
        model: Model used for generation.
        target_fields: Where to write translations. If None, overwrite the
            source fields in place. If given, must match fields in length;
            originals are left untouched.

    Returns:
        An ALTER DataSpec named "alter_translate_to_language".

    Raises:
        ValueError: If fields is empty, or if target_fields is given and its
            length does not match fields. Runtime errors (missing source
            field, wrong response count, empty completions, wrong response
            shape) are raised from _execute and surface through the runner
            as a RuntimeError whose __cause__ is the original exception.
    """
    _validate_args("translate.to_language", fields, target_fields)
    system_message: Message = {
        "role": "system",
        "content": SYSTEM_PROMPT_LANGUAGE.format(target_language=target_language),
    }
    return _make_translate_node(
        source,
        fields=fields,
        model=model,
        target_fields=target_fields,
        system_message=system_message,
        node_name="alter_translate_to_language",
        caller_name="translate.to_language",
        transform=str.strip,
    )


def to_code(
    source: DataSpec,
    *,
    target_language: str,
    fields: list[str],
    model: Model,
    source_language: str | None = None,
    target_fields: list[str] | None = None,
) -> DataSpec:
    """Translate text fields on each item into code in target_language.

    Each source field can be either a natural-language description of what
    the code should do, or existing source code in another language that
    should be ported. Each (item, field) pair is one LLM call, all dispatched
    through a single generate_batch so they run in parallel.

    Args:
        source: Upstream DataSpec providing list[dict] data.
        target_language: Programming language to emit (e.g. "Python", "Rust",
            "TypeScript").
        fields: Keys whose values should be converted to code.
        model: Model used for generation.
        source_language: Optional hint describing the input. When provided,
            the model is told the input is written in this language (e.g.
            "Python", "pseudocode", "natural language"). When None, the
            model infers.
        target_fields: Where to write the generated code. If None, overwrite
            the source fields in place. If given, must match fields in length;
            originals are left untouched.

    Returns:
        An ALTER DataSpec named "alter_translate_to_code".

    Raises:
        ValueError: If fields is empty, or if target_fields is given and its
            length does not match fields. Runtime errors (missing source
            field, wrong response count, empty completions, wrong response
            shape) are raised from _execute and surface through the runner
            as a RuntimeError whose __cause__ is the original exception.
    """
    _validate_args("translate.to_code", fields, target_fields)
    source_hint = (
        f" The input is written in {source_language}."
        if source_language is not None
        else ""
    )
    system_message: Message = {
        "role": "system",
        "content": SYSTEM_PROMPT_CODE.format(
            target_language=target_language,
            source_hint=source_hint,
        ),
    }
    return _make_translate_node(
        source,
        fields=fields,
        model=model,
        target_fields=target_fields,
        system_message=system_message,
        node_name="alter_translate_to_code",
        caller_name="translate.to_code",
        transform=_strip_code_fences,
    )


def _validate_args(
    caller_name: str,
    fields: list[str],
    target_fields: list[str] | None,
) -> None:
    """Check call-time arguments shared by to_language and to_code."""
    if not fields:
        raise ValueError(
            f"{caller_name}: fields must be a non-empty list of field names"
        )
    if target_fields is not None and len(target_fields) != len(fields):
        raise ValueError(
            f"{caller_name}: target_fields length ({len(target_fields)}) "
            f"must match fields length ({len(fields)})"
        )


def _strip_code_fences(text: str) -> str:
    """Strip an outer markdown code fence from a model response.

    Many code models wrap their output in ```lang ... ``` even when told
    not to. This removes the outer fence (with or without a language tag)
    and returns the inner code with surrounding whitespace trimmed. If the
    text has no outer fence it is returned stripped.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    # Drop the opening fence line (```lang or just ```).
    first_newline = stripped.find("\n")
    if first_newline == -1:
        return stripped
    body = stripped[first_newline + 1 :]
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


def _make_translate_node(
    source: DataSpec,
    *,
    fields: list[str],
    model: Model,
    target_fields: list[str] | None,
    system_message: Message,
    node_name: str,
    caller_name: str,
    transform: Callable[[str], str],
) -> DataSpec:
    """Build the ALTER node that does the per-(item, field) generate_batch.

    Shared by to_language and to_code. The system_message, node_name,
    caller_name, and transform change between them; the batching and
    error-handling logic does not.
    """
    write_fields = target_fields if target_fields is not None else fields

    def _execute(data: list[dict]) -> list[dict]:
        if not data:
            return []

        prompts: list[Prompt] = []
        for item in data:
            for field in fields:
                value = get_field(item, field, caller_name)
                user_message: Message = {"role": "user", "content": value}
                prompts.append([system_message, user_message])

        responses = model.generate_batch(prompts, n=1)

        if len(responses) != len(prompts):
            raise ValueError(
                f"{caller_name}: model returned {len(responses)} "
                f"completions for {len(prompts)} prompts"
            )

        results: list[dict] = []
        idx = 0
        for item_idx, item in enumerate(data):
            new_item = dict(item)
            for field, write_key in zip(fields, write_fields):
                choices = responses[idx]
                if not isinstance(choices, list):
                    raise ValueError(
                        f"{caller_name}: expected list of completions "
                        f"for item {item_idx} field {field!r}, got "
                        f"{type(choices).__name__}"
                    )
                if not choices:
                    raise ValueError(
                        f"{caller_name}: model returned no completions "
                        f"for item {item_idx} field {field!r}"
                    )
                new_item[write_key] = transform(choices[0])
                idx += 1
            results.append(new_item)
        return results

    return DataSpec(NodeType.ALTER, source, _execute, node_name)
