"""Self-Instruct pipeline (Wang et al., 2022) in Forger.

Reference: https://arxiv.org/abs/2212.10560
"""
from __future__ import annotations

import json
import os
import random
import re

import pyforger
from pyforger.acquire import Model, from_list
from pyforger.core import DataSpec, NodeType

from examples.self_instruct.prompts import (
    CLASSIFICATION_PROMPT,
    build_input_first_prompt,
    build_instruction_messages,
    build_output_first_prompt,
)


BLOCKLIST = [
    "image", "images", "graph", "graphs", "picture", "pictures", "file", "files",
    "map", "maps", "draw", "plot", "go to", "video", "audio", "music",
    "flowchart", "diagram",
]
MODEL_ID = os.environ.get("FORGER_MODEL_ID", "openai/wine-llama3-2-3b-instruct")
SEED_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "seed_tasks.jsonl")

_TASK_LINE = re.compile(r"Task\s+\d+:\s*(.+)")
_INPUT_RE  = re.compile(r"Input:\s*(.+?)(?=Output:|$)", re.DOTALL)
_OUTPUT_RE = re.compile(r"Output:\s*(.+?)(?=Input:|Example|$)", re.DOTALL)


def parse_task_lines(raws):
    return [
        {"instruction": m.group(1).strip()}
        for r in raws
        for line in r.get("text", "").split("\n")
        if (m := _TASK_LINE.match(line.strip()))
    ]


def parse_instance(raw):
    inp, out = _INPUT_RE.search(raw), _OUTPUT_RE.search(raw)
    return {
        "input":  inp.group(1).strip() if inp else "",
        "output": out.group(1).strip() if out else raw.strip(),
    }


def generate_round(pool, instruction_model):
    human   = [t for t in pool if t["source"] == "human"]
    machine = [t for t in pool if t["source"] == "machine"]
    few_shot = random.sample(human, min(6, len(human)))
    if machine:
        few_shot += random.sample(machine, min(2, len(machine)))

    raw = pyforger.run(pyforger.acquire.sample_from_LLM(
        prompt=build_instruction_messages(few_shot),
        model=instruction_model, n=10,
    ))
    spec = from_list(parse_task_lines(raw), name="parsed_instructions")
    spec = pyforger.filter.by_rouge_l(spec, field="instruction", threshold=0.7)
    spec = pyforger.filter.by_keyword_blocklist(spec, response_field="instruction", blocklist=BLOCKLIST)
    spec = pyforger.filter.by_length(spec, instruction_field="instruction", min_len=3, max_len=150)
    return [{**t, "source": "machine"} for t in pyforger.run(spec)]


def build_pipeline(pool, classification_model, instance_model):
    """Return a `def self_instruct()` closure for `pyforger.run` to compile."""
    human_seeds   = [t for t in pool if t["source"] == "human"]
    clf_seeds     = [t for t in human_seeds if     t.get("is_classification")]
    non_clf_seeds = [t for t in human_seeds if not t.get("is_classification")]

    def _as_source(row, name_):
        """Wrap a single row so generate_from_context sees a list[dict]."""
        if isinstance(row, dict):
            return from_list([row], name=f"{name_}_source")
        return DataSpec(NodeType.ALTER, [row], lambda r: [r], f"{name_}_source")

    def _parse_gen(g):
        return [{**it, **parse_instance(it["text"])} for it in g]

    def _zip_classify(ps, cs):
        return [{**p, **c} for p, c in zip(ps, cs)]

    def generate_instances(row, prompt_builder, seeds, name_):
        gen = pyforger.alter.generate_from_context(
            input=_as_source(row, name_), model=instance_model,
            temperature=0.0, samples_per_item=1,
            prompt_fn=lambda t: prompt_builder(t["instruction"], seeds),
            name=name_,
        )
        return DataSpec(NodeType.ALTER, [gen], _parse_gen, f"{name_}_parse")

    def self_instruct():
        pool_ds = from_list(pool, name="instruction_pool")
        classify_out = pyforger.alter.generate_from_context(
            input=pool_ds, model=classification_model,
            temperature=0.0, samples_per_item=1,
            prompt_fn=lambda t: CLASSIFICATION_PROMPT.replace("{value}", t["instruction"]),
            name="classify",
        )
        # generate_from_context returns only {"text": ...}; re-zip with
        # the pool so downstream rows keep "instruction" and friends.
        instruction_types = DataSpec(
            NodeType.ALTER, [pool_ds, classify_out], _zip_classify, "classify_with_context",
        )

        instance_branches = []
        for t in instruction_types:
            if t["text"] == "Yes":
                instance_branches.append(generate_instances(
                    t, build_output_first_prompt, clf_seeds, "output_first_gen"))
            else:
                instance_branches.append(generate_instances(
                    t, build_input_first_prompt, non_clf_seeds, "input_first_gen"))

        spec = pyforger.alter.concat(instance_branches, name="all_instances")
        spec = pyforger.filter.by_exact_match(spec, field="input")
        spec = pyforger.filter.by_invalid_response(spec, instruction_field="instruction", response_field="output")
        spec = pyforger.filter.by_length(spec, instruction_field="output", min_len=1, max_len=150)
        return spec

    return self_instruct


def run_pipeline():
    common = dict(api_key=os.environ["LITELLM_API_KEY"], api_base=os.environ["LITELLM_BASE_URL"])
    instruction_model    = Model(MODEL_ID, temperature=0.7, max_tokens=1024, **common)
    classification_model = Model(MODEL_ID, temperature=0.0, max_tokens=3,    **common)
    instance_model       = Model(MODEL_ID, temperature=0.0, max_tokens=300,  **common)

    with open(SEED_PATH) as f:
        pool = [{**json.loads(line), "source": "human"} for line in f]
    print(f"Step 1: loaded {len(pool)} seed tasks")

    new = generate_round(pool, instruction_model)
    pool.extend(new)
    print(f"Step 2 round 1: +{len(new)} (pool={len(pool)})")

    return pyforger.run(build_pipeline(pool, classification_model, instance_model))


if __name__ == "__main__":
    result = run_pipeline()
    print(f"Done! Generated {len(result)} instruction-tuning instances")
