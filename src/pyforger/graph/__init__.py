"""Graph-capture layer for Forger.

Enables fully-lazy pipelines with control flow. Users write:

    def my_pipeline():
        ...
        instances = []
        for t in instruction_types:        # instruction_types is a DataSpec
            if t["text"].startswith("Yes"):
                instances.append(generate_instances(t, ...))
            else:
                instances.append(generate_instances(t, ...))
        return pyforger.filter.by_length(pyforger.alter.concat(instances), ...)

    result = pyforger.run(my_pipeline)

Under the hood `run(fn)` parses the function's AST, compiles it into a
DataSpec graph with FOREACH / COND nodes, and executes lazily. No LLM
calls fire until `run()` is invoked.
"""
from pyforger.graph.compiler import compile_function
from pyforger.graph.primitives import cond, foreach
from pyforger.graph.symbolic import ExprSpec, ItemSpec, TracingError

__all__ = [
    "compile_function",
    "cond",
    "foreach",
    "ExprSpec",
    "ItemSpec",
    "TracingError",
]
