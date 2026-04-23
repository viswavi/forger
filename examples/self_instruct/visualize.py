"""Visualize the fully-lazy Self-Instruct routing graph — no LLM calls.

Run: python -m examples.self_instruct.visualize

Uses `build_pipeline` with stub models. Importing pipeline.py doesn't
fire any LLM work thanks to the factory. Compilation is pure AST work —
you can render the graph without network access.
"""
from __future__ import annotations

import os

from pyforger import viz
from pyforger.graph.compiler import compile_function

from examples.self_instruct.pipeline import build_pipeline

HERE = os.path.dirname(__file__)


class _StubModel:
    """A do-nothing stand-in for Model. compile_function never calls it."""

    def generate(self, *a, **kw):  # noqa: ARG002
        raise RuntimeError("stub model — compilation shouldn't invoke it")

    def generate_batch(self, *a, **kw):  # noqa: ARG002
        raise RuntimeError("stub model — compilation shouldn't invoke it")


stub_pool = [
    {"instruction": "Classify this review.", "source": "human", "is_classification": True},
    {"instruction": "Explain photosynthesis.", "source": "human", "is_classification": False},
]
self_instruct = build_pipeline(stub_pool, _StubModel(), _StubModel())
spec = compile_function(self_instruct)

print("=== ASCII tree ===")
viz.print_tree(spec)
print()
print("=== Summary ===")
print(viz.summary(spec))
print()

mmd_path = os.path.join(HERE, "graph.mmd")
with open(mmd_path, "w") as f:
    f.write(viz.to_mermaid(spec))
print(f"Wrote Mermaid: {mmd_path}")

dot_path = os.path.join(HERE, "graph.dot")
with open(dot_path, "w") as f:
    f.write(viz.to_dot(spec))
print(f"Wrote DOT:     {dot_path}")
