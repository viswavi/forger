"""Label-based routing decompose operations for Forger.

Classify each upstream item with an LLM, then route items sharing a label
to a per-label sub-pipeline and concatenate the sub-pipeline outputs.
"""
from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Any, Union

from pyforger.acquire import from_list
from pyforger.acquire.model import Model, Prompt
from pyforger.core import DataSpec, NodeType

PromptTemplate = Union[str, Callable[[dict], Prompt]]


def _resolve_prompt(template: PromptTemplate, item: dict) -> Prompt:
    """Resolve a prompt template against one input item.

    A string template is filled from the item's keys via ``str.format_map``.
    A callable is invoked with the item.
    """
    if isinstance(template, str):
        try:
            return template.format_map(item)
        except KeyError as e:
            raise ValueError(
                f"prompt_template references field {e.args[0]!r} "
                f"missing from item keys {list(item)}"
            ) from None
    return template(item)


def _require_hashable(label: Any, raw: str, i: int) -> None:
    """Raise TypeError with a helpful message if label is not hashable."""
    try:
        hash(label)
    except TypeError as e:
        raise TypeError(
            f"parse_fn returned unhashable label {label!r} "
            f"for response {raw!r} at index {i}. Labels must be "
            "hashable (used as dict keys)."
        ) from e


def by_llm_label(
    source: DataSpec,
    *,
    model: Model,
    prompt_template: PromptTemplate,
    labels: dict[Hashable, Callable[[DataSpec], DataSpec]],
    parse_fn: Callable[[str], Hashable] = lambda r: r.strip().lower(),
    default: Callable[[DataSpec], DataSpec] | None = None,
    temperature: float | None = None,
    name: str = "by_llm_label",
) -> DataSpec:
    """Classify each item with an LLM, then route to labels[label].

    For each upstream item, prompt_template is resolved and all prompts are
    sent to model.generate_batch in one call. parse_fn converts each
    response to a label value. Items sharing a label are collected and
    passed as a source DataSpec to labels[label](bucket_source); the
    sub-pipeline's output is concatenated into the final result in
    first-seen-label order. Items whose label is not in labels go through
    default (when set) or are dropped.

    Each sub-pipeline is executed via a nested ``run()`` with its own
    cache. Results are not shared with the outer graph's cache, so a
    DataSpec referenced from both a sub-pipeline and the outer graph will
    execute twice.

    Args:
        source: Upstream DataSpec providing list[dict] data.
        model: Model instance used for classification.
        prompt_template: Either a format string filled per item via
            ``str.format_map`` (use ``{{`` / ``}}`` to produce literal
            braces) or a callable ``(dict) -> Prompt``.
        labels: Mapping from parsed label value to a sub-pipeline builder.
            The builder takes a DataSpec (bucket of items sharing that
            label) and returns a DataSpec whose output is included in the
            final result. Label values must be hashable; note that
            ``True == 1`` under Python hashing, so mixing bool and int
            keys is ambiguous.
        parse_fn: Converts the raw LLM response to a label value looked up
            in labels. Default strips whitespace and lowercases so label
            keys should be lowercase; override for case-sensitive or
            structured parsing.
        default: Optional sub-pipeline builder for items whose parsed
            label is not in labels. If None, such items are dropped.
        temperature: Sampling temperature passed to model.generate_batch.
            None (default) uses the model's configured temperature. Pass a
            float to override per call.
        name: Label for the DataSpec node. The outer node is named
            ``f"decompose_{name}"`` and each bucket source is named
            ``f"{name}_bucket_{i}_{label}"``.

    Returns:
        A DECOMPOSE DataSpec whose output is the concatenation of each
        sub-pipeline's output, in first-seen-label order.
    """
    from pyforger.runner import run as _run

    def _execute(data: list[dict]) -> list[dict]:
        if not data:
            return []

        prompts: list[Prompt] = [
            _resolve_prompt(prompt_template, item) for item in data
        ]
        gen_kwargs: dict[str, Any] = {}
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        responses = model.generate_batch(prompts, n=1, **gen_kwargs)
        if len(responses) != len(prompts):
            raise RuntimeError(
                f"model.generate_batch returned {len(responses)} responses "
                f"for {len(prompts)} prompts. Cannot route items safely."
            )

        parsed_labels: list[Hashable] = []
        for i, response in enumerate(responses):
            raw = response[0]
            label = parse_fn(raw)
            _require_hashable(label, raw, i)
            parsed_labels.append(label)

        buckets: dict[Hashable, list[dict]] = {}
        order: list[Hashable] = []
        for item, label in zip(data, parsed_labels):
            if label not in buckets:
                buckets[label] = []
                order.append(label)
            buckets[label].append(item)

        result: list[dict] = []
        for i, label in enumerate(order):
            items = buckets[label]
            builder = labels[label] if label in labels else default
            if builder is None:
                continue
            bucket_src = from_list(items, name=f"{name}_bucket_{i}_{str(label)[:40]}")
            sub_pipeline = builder(bucket_src)
            result.extend(_run(sub_pipeline))
        return result

    return DataSpec(NodeType.DECOMPOSE, source, _execute, f"decompose_{name}")
