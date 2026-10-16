"""EvalTree capability trees and weakness profiles (arXiv:2503.08893).

Adapted from Zhiyuan Zeng et al.'s MIT-licensed reference implementation:
https://github.com/zhiyuan-zeng/EvalTree (copyright 2025 Zhiyuan Zeng).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from pyforger.acquire.model import Model, Prompt
from pyforger.inspect.cluster import _default_embed
from pyforger.op import op
from pyforger.utils import get_field

_ANNOTATE_SYSTEM = (
    "Given a coding problem, describe the specific coding capability needed to "
    "solve it. Return only one detailed, action-oriented gerund phrase. "
    "Describe the skill, without copying details from the problem. "
    "Do not judge whether a model's answer is correct."
)
_DESCRIBE_SYSTEM = (
    "Summarize the coding capabilities below as one informative phrase covering "
    "the skills in this group. Do not introduce unrelated skills. "
    "Return only the phrase."
)


@op(name="inspect.eval_tree")
def eval_tree(
    items: list[dict],
    *,
    input_field: str,
    score_field: str,
    model: Model,
    reference_field: str | None = None,
    embed_fn: Callable[[list[str]], Any] | None = None,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    max_children: int = 10,
    threshold: float = 0.5,
    alpha: float = 0.05,
    min_ci_size: int = 5,
    min_child_size: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """Build an EvalTree report from scored model outputs.

    ``score_field`` must contain a boolean or binary 0/1 correctness score.
    ``input_field`` holds the problem; optional ``reference_field`` holds a
    *verified* solution to improve capability annotation. Generated model
    outputs are deliberately not used as the reference. ``embed_fn`` receives
    the capability phrases and returns one vector per phrase; by default the
    same local sentence-transformers backend as topic_based_clustering is used.

    Returns ``{"tree": ..., "weaknesses": ..., "capabilities": ...}``.
    Tree nodes contain row indices, accuracy, a Clopper-Pearson interval where
    eligible, and child nodes. A weakness is the highest eligible subtree whose
    confidence interval lies entirely below ``threshold``, unless a child of
    at least ``min_child_size`` does not satisfy that test (following EvalTree).
    """
    if max_children < 2:
        raise ValueError("max_children must be at least 2")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if min_ci_size < 1 or min_child_size < 1:
        raise ValueError("min_ci_size and min_child_size must be positive")
    if not items:
        return {"tree": None, "weaknesses": [], "capabilities": []}

    inputs = [str(get_field(row, input_field, "eval_tree")) for row in items]
    references = (
        [str(get_field(row, reference_field, "eval_tree")) for row in items]
        if reference_field is not None
        else None
    )
    scores = [get_field(row, score_field, "eval_tree") for row in items]
    if any(score not in (0, 1, False, True) for score in scores):
        raise ValueError("eval_tree: score_field must contain only booleans or 0/1")
    scores = [int(score) for score in scores]

    prompts: list[Prompt] = []
    for i, problem in enumerate(inputs):
        content = f"Problem:\n{problem}"
        if references is not None:
            content += f"\n\nVerified solution:\n{references[i]}"
        prompts.append(
            [
                {"role": "system", "content": _ANNOTATE_SYSTEM},
                {"role": "user", "content": content},
            ]
        )
    responses = model.generate_batch(prompts, temperature=0.0)
    capabilities = [
        _phrase(response[0], "capability annotation") for response in responses
    ]

    vectors = (
        embed_fn(
            [
                "The model has the following skill or capability: " + c
                for c in capabilities
            ]
        )
        if embed_fn is not None
        else _default_embed(
            [
                "The model has the following skill or capability: " + c
                for c in capabilities
            ],
            embedding_model,
        )
    )
    try:
        import numpy as np
        from scipy.stats import beta
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError as exc:
        raise ImportError("eval_tree requires pyforger[clustering]") from exc

    matrix = np.asarray(vectors, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != len(items)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("eval_tree: embed_fn must return one finite vector per row")

    def make_node(indices: list[int]) -> dict[str, Any]:
        count = len(indices)
        correct = sum(scores[i] for i in indices)
        interval = None
        if count >= min_ci_size:
            # Exact binomial interval, matching statsmodels' method="beta".
            low = (
                0.0
                if correct == 0
                else float(beta.ppf(alpha / 2, correct, count - correct + 1))
            )
            high = (
                1.0
                if correct == count
                else float(beta.ppf(1 - alpha / 2, correct + 1, count - correct))
            )
            interval = [low, high]
        return {
            "description": None,
            "indices": indices,
            "size": count,
            "correct": correct,
            "accuracy": correct / count,
            "confidence_interval": interval,
            "children": [],
        }

    def build(indices: list[int]) -> dict[str, Any]:
        node = make_node(indices)
        if len(indices) == 1:
            return node
        if len(indices) == 2:
            groups = [[i] for i in indices]
        else:
            groups = []
            best_score = 0.0
            points = matrix[indices]
            distinct_points = len(np.unique(points, axis=0))
            for k in range(2, min(max_children, len(indices) - 1, distinct_points) + 1):
                labels = KMeans(
                    n_clusters=k, init="k-means++", random_state=seed, n_init=10
                ).fit_predict(points)
                distinct = len(set(labels))
                if distinct < 2 or distinct >= len(indices):
                    continue
                try:
                    score = float(silhouette_score(points, labels, metric="cosine"))
                except ValueError:  # e.g. zero vectors cannot use cosine distance
                    continue
                if math.isfinite(score) and score > best_score:
                    best_score = score
                    groups = [
                        [
                            index
                            for index, label in zip(indices, labels)
                            if label == group
                        ]
                        for group in range(k)
                        if group in labels
                    ]
            if not groups:
                groups = [[i] for i in indices]
        node["children"] = [build(group) for group in groups]
        return node

    tree = build(list(range(len(items))))

    # Bottom-up descriptions. Batch nodes at each depth, as children must be
    # described before their parents and model.generate_batch preserves order.
    levels: dict[int, list[dict[str, Any]]] = {}

    def collect(node: dict[str, Any], depth: int) -> None:
        levels.setdefault(depth, []).append(node)
        for child in node["children"]:
            collect(child, depth + 1)

    collect(tree, 0)
    for depth in sorted(levels, reverse=True):
        internal = [node for node in levels[depth] if node["children"]]
        for node in levels[depth]:
            if not node["children"]:
                node["description"] = capabilities[node["indices"][0]]
        if internal:
            prompts = [
                [
                    {"role": "system", "content": _DESCRIBE_SYSTEM},
                    {
                        "role": "user",
                        "content": "Skill descriptions:\n"
                        + "\n".join(
                            f"{i + 1}. {child['description']}"
                            for i, child in enumerate(node["children"])
                        ),
                    },
                ]
                for node in internal
            ]
            responses = model.generate_batch(prompts, temperature=0.0)
            for node, response in zip(internal, responses):
                node["description"] = _phrase(response[0], "capability description")

    weaknesses: list[dict[str, Any]] = []

    def extract(node: dict[str, Any]) -> None:
        interval = node["confidence_interval"]
        eligible = interval is not None and interval[1] < threshold
        large_child_fails = any(
            child["size"] >= min_child_size
            and (
                child["confidence_interval"] is None
                or child["confidence_interval"][1] >= threshold
            )
            for child in node["children"]
        )
        if eligible and not large_child_fails:
            weaknesses.append(
                {
                    "capability": node["description"],
                    "indices": node["indices"],
                    "size": node["size"],
                    "accuracy": node["accuracy"],
                    "confidence_interval": interval,
                }
            )
        else:
            for child in node["children"]:
                extract(child)

    extract(tree)
    return {"tree": tree, "weaknesses": weaknesses, "capabilities": capabilities}


def _phrase(response: str, stage: str) -> str:
    phrase = response.strip()
    if not phrase:
        raise RuntimeError(f"eval_tree: empty {stage} response")
    return phrase
