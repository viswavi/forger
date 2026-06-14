"""ICL prompt construction and code-completion sanitization.

- :func:`sample_subsets` — seeded random k-subsets of the demo pool.
- :func:`build_messages` — multi-turn ``[user, assistant, ..., user]`` prompt
  (matches the TRL shape of the Forger data).
- :func:`extract_code` — strip the first fenced block from a completion.
"""
from __future__ import annotations

import random
import re
from collections.abc import Sequence

Message = dict[str, str]


def sample_subsets(
    pool: Sequence[dict],
    *,
    k: int,
    n_subsets: int,
    seed: int,
) -> list[list[dict]]:
    """Draw ``n_subsets`` seeded k-subsets from ``pool``.

    Within a subset items are sampled without replacement; across subsets,
    draws are independent (subsets may overlap). Callers typically derive
    a per-problem seed from a base seed so each problem draws independently.
    Raises :class:`ValueError` if ``k`` or ``n_subsets`` is invalid.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if n_subsets < 0:
        raise ValueError(f"n_subsets must be non-negative, got {n_subsets}")
    if k > len(pool):
        raise ValueError(
            f"k ({k}) exceeds pool size ({len(pool)}); "
            "use a smaller k or a larger demo pool"
        )
    rng = random.Random(seed)
    return [rng.sample(list(pool), k) for _ in range(n_subsets)]


def build_messages(demos: Sequence[dict], target_prompt: str) -> list[Message]:
    """Assemble a multi-turn chat prompt with optional few-shot demos.

    Empty ``demos`` produces a single user turn (the zero-shot baseline);
    otherwise the messages alternate user/assistant per demo and end on a
    user turn carrying ``target_prompt``. No system message is added.
    """
    messages: list[Message] = []
    for demo in demos:
        try:
            messages.append({"role": "user", "content": demo["prompt"]})
            messages.append({"role": "assistant", "content": demo["solution"]})
        except KeyError as exc:
            raise KeyError(
                f"build_messages: demo is missing required field {exc.args[0]!r}; "
                f"got keys {list(demo.keys())}"
            ) from exc
    messages.append({"role": "user", "content": target_prompt})
    return messages


_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)


def extract_code(prompt: str, completion: str) -> str:
    """Return the runnable code from a model's completion.

    If a ```` ```python ```` (or bare triple-backtick) fenced block is
    present, returns the contents of the first one; otherwise returns the
    completion unchanged. Re-stated ``def`` signatures are left intact;
    Python's def-shadowing handles them and stripping would discard helper
    functions emitted before the target. ``prompt`` is reserved for future
    signature-aware sanitization.
    """
    del prompt

    text = completion
    fence_match = _FENCE_RE.search(text)
    if fence_match is not None:
        text = fence_match.group(1)

    if text.startswith("\n"):
        text = text[1:]
    return text
