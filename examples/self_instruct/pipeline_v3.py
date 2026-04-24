"""Self-Instruct (Wang et al., 2022) on the Dask-style lazy runtime.

What this demonstrates
----------------------
  * Ops return Lazy[T]. Chains stay lazy — no DataSpec, no AST compiler.
  * Python control flow (``for``, ``if``, list comp) materializes lazily
    upstream at the boundary. The classify+dispatch in Step 3/4 is a
    list-comprehension partition; no decompose.by_llm_label.
  * Caching is content-addressed at the op boundary. Re-running the
    script short-circuits every op whose inputs haven't changed.
  * Offline-friendly: uses StubModel so the shape of the pipeline is
    inspectable with no API keys. Swap in a real Model to run for real.

Compare to pipeline.py (DataSpec / lazy via AST) and pipeline_v2.py (eager
@op prototype). This is the Dask shape: lazy until a materialization
boundary, with plain-Python control flow.
"""
from __future__ import annotations

import re

from pyforger.lazy import acquire, alter, clear_trace, filter, op, run, trace


# ---------------------------------------------------------------------------
# Stub model — deterministic, offline. Swap for a real Model at call sites.
# ---------------------------------------------------------------------------

class StubModel:
    """Offline stand-in for an LLM. Pattern-matches the prompt text."""

    def __init__(self, kind: str) -> None:
        self.kind = kind  # "classifier" or "generator"

    def cache_key(self) -> str:
        return f"StubModel({self.kind})"

    def generate_batch(self, prompts: list[str], temperature: float = 0.0) -> list[list[str]]:
        out = []
        for p in prompts:
            if self.kind == "classifier":
                is_clf = any(w in p.lower() for w in ("classify", "category", "label", "sentiment"))
                out.append(["Yes" if is_clf else "No"])
            else:
                out.append([f"Input: sample input\nOutput: sample output for task: {p[-40:].strip()}"])
        return out


classifier_model = StubModel("classifier")
instance_model = StubModel("generator")


# ---------------------------------------------------------------------------
# Seeds. Hardcoded so this runs anywhere.
# ---------------------------------------------------------------------------

SEED_TASKS: list[dict] = [
    {"instruction": "Classify the sentiment of this review as positive or negative.", "is_classification": True},
    {"instruction": "Label the following sentence with its topic category.",          "is_classification": True},
    {"instruction": "Write a short poem about autumn.",                               "is_classification": False},
    {"instruction": "Translate this sentence into French.",                           "is_classification": False},
    {"instruction": "Solve the equation 2x + 3 = 11.",                                "is_classification": False},
    {"instruction": "Classify whether this tweet is spam.",                           "is_classification": True},
    {"instruction": "Summarize the following paragraph in one sentence.",             "is_classification": False},
    {"instruction": "Label the genre of this movie description.",                     "is_classification": True},
]


# ---------------------------------------------------------------------------
# Helpers — plain Python. Prompt builders, parsers, predicates.
# ---------------------------------------------------------------------------

def classification_prompt(item: dict) -> str:
    return (
        f"Does the following task ask for classification? "
        f"Answer Yes or No.\n\nTask: {item['instruction']}"
    )


def output_first_prompt(item: dict) -> str:
    return f"Generate Input/Output for this classify-style task:\n{item['instruction']}"


def input_first_prompt(item: dict) -> str:
    return f"Generate Input/Output for this open-ended task:\n{item['instruction']}"


def is_classification(label_row: dict) -> bool:
    return label_row["text"].strip().lower().startswith("yes")


def parse_instance(raw: str) -> dict[str, str]:
    inp = re.search(r"Input:\s*(.+?)(?=Output:|$)", raw, re.DOTALL)
    out = re.search(r"Output:\s*(.+?)(?=Input:|$)", raw, re.DOTALL)
    return {
        "input":  inp.group(1).strip() if inp else "",
        "output": out.group(1).strip() if out else raw.strip(),
    }


@op(name="local.parse_instances")
def parse_instances(rows: list[dict]) -> list[dict]:
    """Wrap parse_instance as an op so it participates in the cache."""
    return [parse_instance(r["text"]) for r in rows]


# ---------------------------------------------------------------------------
# Pipeline. Nothing executes here — just graph construction, except at the
# list-comprehension boundary where `labels` is forced.
# ---------------------------------------------------------------------------

def build_pipeline():
    pool = acquire.from_list(SEED_TASKS)

    # Step 3: classify each instruction. Stays Lazy.
    labels = alter.generate_from_context(
        pool, model=classifier_model, prompt_fn=classification_prompt,
    )

    # Step 4: dispatch. `zip(pool, labels)` iterates both Lazies, forcing
    # materialization of the classification subgraph. From this line until
    # the next Forger op we're in plain Python with concrete lists — which
    # is exactly where data-dependent branching belongs.
    clf_items     = [it for it, lbl in zip(pool, labels) if     is_classification(lbl)]
    non_clf_items = [it for it, lbl in zip(pool, labels) if not is_classification(lbl)]
    print(f"  dispatch: {len(clf_items)} classification, {len(non_clf_items)} non-classification")

    # Back to Lazy. Each branch generates instances with its own prompt.
    clf_gen = alter.generate_from_context(
        clf_items, model=instance_model, prompt_fn=output_first_prompt,
    )
    non_clf_gen = alter.generate_from_context(
        non_clf_items, model=instance_model, prompt_fn=input_first_prompt,
    )

    # Concatenate and parse — still lazy.
    merged = alter.concat(clf_gen, non_clf_gen)
    parsed = parse_instances(merged)

    # Step 5: filter chain — all lazy, no intermediate materialization.
    parsed = filter.by_invalid_response(
        parsed, instruction_field="input", response_field="output",
    )
    parsed = filter.by_length(parsed, field="output", min_len=1, max_len=150)
    return parsed


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    clear_trace()

    print("Building pipeline graph (no LLM calls yet)...")
    pipeline = build_pipeline()
    print(f"Final node: {pipeline!r}\n")

    print("Execution plan (pre-compute):")
    print(pipeline.plan())
    print()

    print("Running...")
    result = run(pipeline)
    print(f"Result: {len(result)} instances")
    for row in result[:3]:
        print(f"  {row}")
    if len(result) > 3:
        print(f"  ... and {len(result) - 3} more")

    print("\nOp trace:")
    for rec in trace():
        tag = "HIT " if rec.cache_hit else "MISS"
        print(f"  {tag} {rec.op_name:40s} {rec.input_hash[:10]} ({rec.duration_s * 1000:.1f}ms)")
