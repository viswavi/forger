"""Prompt builder shared by split construction, evaluation and training data."""
from __future__ import annotations


def build_candidate_prompt(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """The chat a model answers: the row's messages as written, with no system prompt.

    Models are evaluated with their default system settings, so nothing is added.
    E.g. [{"role": "user", "content": "48M with new AF, CHA2DS2-VASc 1 ..."}].
    """
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("a prompt must end with a user message")
    return [{"role": m["role"], "content": m["content"]} for m in messages]
