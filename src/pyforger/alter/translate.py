"""Translation alter operations for Forger.

Uses an LLM (via pyforger.acquire.model.Model) to translate text fields on
each item of a list[dict] into a target natural language. Each (item, field)
pair is translated in its own LLM call; calls across items and fields are
dispatched in a single generate_batch so litellm can run them in parallel.
"""
from __future__ import annotations

from pyforger.acquire.model import Message, Model, Prompt
from pyforger.core import DataSpec, NodeType
from pyforger.utils import get_field

SYSTEM_PROMPT = (
    "You are a professional translator. Translate the user's text into "
    "{target_language}. Output only the translated text with no preamble, "
    "commentary, or quotation marks. Preserve formatting such as code "
    "blocks, markdown, placeholders, and URLs unchanged."
)


def to_language(
    source: DataSpec,
    *,
    target_language: str,
    fields: list[str],
    model: Model,
    target_fields: list[str] | None = None,
) -> DataSpec:
    """Translate text fields on each item into target_language using an LLM.

    Each field on each item is translated in its own LLM call. All calls for
    one pipeline run are dispatched as a single generate_batch so the
    underlying LiteLLM backend can run them in parallel.

    Args:
        source: Upstream DataSpec providing list[dict] data.
        target_language: Free-form language name passed to the model
            (e.g. "Spanish", "Brazilian Portuguese", "zh-Hans").
        fields: Keys in each item whose values should be translated.
        model: Caller-provided Model used for generation.
        target_fields: Where to write translations. If None, translations
            overwrite the source fields in place. If provided, must be the
            same length as fields; source fields are left untouched.

    Returns:
        An ALTER DataSpec named "alter_translate_to_language".

    Raises:
        ValueError: At call time if fields is empty, or if target_fields is
            provided and its length does not match fields.

    Note:
        Run-time errors (missing source fields, wrong response count, empty
        completions, wrong response shape) are raised from inside the node's
        execute function and wrapped by pyforger.runner into a RuntimeError.
        The original exception is available on ``RuntimeError.__cause__``.
    """
    if not fields:
        raise ValueError(
            "translate.to_language: fields must be a non-empty list of field names"
        )
    if target_fields is not None and len(target_fields) != len(fields):
        raise ValueError(
            f"translate.to_language: target_fields length "
            f"({len(target_fields)}) must match fields length ({len(fields)})"
        )

    write_fields = target_fields if target_fields is not None else fields
    system_message: Message = {
        "role": "system",
        "content": SYSTEM_PROMPT.format(target_language=target_language),
    }

    def _execute(data: list[dict]) -> list[dict]:
        if not data:
            return []

        prompts: list[Prompt] = []
        for item in data:
            for field in fields:
                value = get_field(item, field, "translate.to_language")
                user_message: Message = {"role": "user", "content": value}
                prompts.append([system_message, user_message])

        responses = model.generate_batch(prompts, n=1)

        if len(responses) != len(prompts):
            raise ValueError(
                f"translate.to_language: model returned {len(responses)} "
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
                        f"translate.to_language: expected list of completions "
                        f"for item {item_idx} field {field!r}, got "
                        f"{type(choices).__name__}"
                    )
                if not choices:
                    raise ValueError(
                        f"translate.to_language: model returned no "
                        f"completions for item {item_idx} field {field!r}"
                    )
                new_item[write_key] = choices[0].strip()
                idx += 1
            results.append(new_item)
        return results

    return DataSpec(NodeType.ALTER, source, _execute, "alter_translate_to_language")
