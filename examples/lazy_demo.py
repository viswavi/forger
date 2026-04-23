"""Minimal demo of fully-lazy for/if/return compiled via AST."""
import pyforger
from pyforger.acquire import from_list


def pipeline():
    source = from_list(
        [{"label": "cat", "n": 1},
         {"label": "dog", "n": 2},
         {"label": "cat", "n": 3}],
        name="input",
    )
    out = []
    for row in source:
        if row["label"] == "cat":
            out.append(_double(row))
        else:
            out.append(_triple(row))
    return pyforger.alter.concat(out, name="merged")


def _double(row):
    from pyforger.core import DataSpec, NodeType
    return DataSpec(NodeType.ALTER, [row], lambda r: [{**r, "v": r["n"] * 2}],
                    "double")


def _triple(row):
    from pyforger.core import DataSpec, NodeType
    return DataSpec(NodeType.ALTER, [row], lambda r: [{**r, "v": r["n"] * 3}],
                    "triple")


if __name__ == "__main__":
    result = pyforger.run(pipeline)
    print("RESULT:", result)
