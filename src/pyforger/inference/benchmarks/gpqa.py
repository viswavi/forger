"""GPQA (graduate-level science MCQ) loader, prompt formatter, and parser.

GPQA is a multiple-choice benchmark of graduate-level physics, chemistry,
and biology questions. This module exposes:

- :class:`GPQAExample` — a single MCQ with shuffled choices and a gold
  letter.
- :func:`load_gpqa` — ``@op``-cached loader that pulls rows from the HF
  hub via :func:`pyforger.acquire.retrieve_from_hf` and converts them to
  ``GPQAExample`` records with a deterministic per-row choice shuffle.
- :func:`format_prompt` / :func:`parse_answer` — the prompt-rendering
  and answer-extraction primitives the eval loop uses.

The dataset (``Idavidrein/gpqa``) is gated; users must accept the license
on the Hub and set ``HF_TOKEN`` (or otherwise authenticate to
``huggingface_hub``) before the first cold cache miss.
"""
from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

from pyforger.acquire.retrieve import retrieve_from_hf
from pyforger.acquire.sources import RetrieveMode
from pyforger.op import op
from pyforger.utils import get_field


@dataclass(frozen=True)
class GPQAExample:
    """One GPQA question with deterministically-shuffled choices.

    Attributes:
        question: The question stem.
        choices: Four answer strings, already shuffled, in A/B/C/D order.
        gold: The correct letter, one of ``"A"``, ``"B"``, ``"C"``,
            ``"D"`` — pointing into ``choices``.
    """

    question: str
    choices: tuple[str, str, str, str]
    gold: str


def _row_to_example(row: dict, idx: int) -> GPQAExample:
    """Convert one HF row to a :class:`GPQAExample`.

    Expected columns (shared across ``gpqa_main``, ``gpqa_diamond``, and
    ``gpqa_extended``): ``"Question"``, ``"Correct Answer"``,
    ``"Incorrect Answer 1"``, ``"Incorrect Answer 2"``,
    ``"Incorrect Answer 3"``.

    Uses :func:`pyforger.utils.get_field` so a future schema change fails
    loudly (``KeyError`` listing the available keys) instead of silently
    producing the catastrophic ~25% accuracy failure mode you'd get from
    misaligning gold answers.

    The choices are shuffled by a deterministic per-row ``random.Random(idx)``
    so position-A is not always correct, but the same row index always
    produces the same shuffle.
    """
    question = get_field(row, "Question", caller="_row_to_example")
    correct = get_field(row, "Correct Answer", caller="_row_to_example")
    incorrect = [
        get_field(row, "Incorrect Answer 1", caller="_row_to_example"),
        get_field(row, "Incorrect Answer 2", caller="_row_to_example"),
        get_field(row, "Incorrect Answer 3", caller="_row_to_example"),
    ]
    choices = [correct, *incorrect]
    order = list(range(4))
    random.Random(idx).shuffle(order)
    shuffled = tuple(choices[j] for j in order)
    gold_idx = order.index(0)
    return GPQAExample(
        question=question,
        choices=shuffled,  # type: ignore[arg-type]
        gold="ABCD"[gold_idx],
    )


@op
def load_gpqa(
    subset: str = "gpqa_main", n: int = 200, seed: int = 0
) -> list[GPQAExample]:
    """Load and shuffle GPQA examples from the Hugging Face Hub.

    Cached on disk via ``@op``; first call pays the HF download cost,
    subsequent calls with the same arguments are a cache hit.

    Args:
        subset: HF config name. One of ``"gpqa_main"``,
            ``"gpqa_diamond"``, or ``"gpqa_extended"``.
        n: Number of examples to retrieve.
        seed: Seed for HF's row-level sampling. The per-example choice
            shuffle is independent of this seed (it's keyed by row index)
            so changing ``seed`` reshuffles which rows you get, not the
            choice order within a row.

    Returns:
        List of :class:`GPQAExample` records.

    Note:
        ``Idavidrein/gpqa`` is a gated dataset. Set ``HF_TOKEN`` in your
        environment (or run ``huggingface-cli login``) and accept the
        dataset license on the Hub before calling this function.
    """
    rows = retrieve_from_hf(
        dataset="Idavidrein/gpqa",
        config=subset,
        split="train",
        mode=RetrieveMode.RANDOM,
        n=n,
        seed=seed,
    )
    return [_row_to_example(row, i) for i, row in enumerate(rows)]


def _render(ex: GPQAExample, *, include_answer: bool) -> str:
    """Render one example as a question block, optionally with the gold answer.

    When ``include_answer`` is True the rendered block ends with
    ``"Answer: <gold>"``, suitable for use as a few-shot exemplar.
    """
    body = (
        f"Question: {ex.question}\n"
        f"A) {ex.choices[0]}\n"
        f"B) {ex.choices[1]}\n"
        f"C) {ex.choices[2]}\n"
        f"D) {ex.choices[3]}\n"
        "Think step by step, then answer with 'Answer: X' where X is A, B, C, or D."
    )
    if include_answer:
        body += f"\nAnswer: {ex.gold}"
    return body


def format_prompt(
    ex: GPQAExample, few_shot: Sequence[GPQAExample] = ()
) -> str:
    """Format an example into a prompt string with optional few-shot exemplars.

    Args:
        ex: The example to predict on (rendered without the answer).
        few_shot: Optional sequence of exemplars rendered before ``ex``,
            each with its gold answer included.

    Returns:
        The full prompt string, with exemplars separated from the target
        by a blank line.
    """
    parts = [_render(shot, include_answer=True) for shot in few_shot]
    parts.append(_render(ex, include_answer=False))
    return "\n\n".join(parts)


_ANSWER_RE = re.compile(r"Answer:\s*\(?([ABCD])\)?", re.IGNORECASE)
# Word-boundary anchored so we don't match letters embedded in English
# words (e.g. the 'd' in "words" or the 'a' in "at all").
_LETTER_FALLBACK_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def parse_answer(completion: str) -> str | None:
    """Extract the predicted letter from a model completion.

    Tries the explicit ``"Answer: X"`` (or ``"Answer: (X)"``) pattern
    first; if that fails, falls back to the last A/B/C/D-like token in
    the string. Returns ``None`` if no letter is found.
    """
    match = _ANSWER_RE.search(completion)
    if match:
        return match.group(1).upper()
    letters = _LETTER_FALLBACK_RE.findall(completion)
    return letters[-1].upper() if letters else None
