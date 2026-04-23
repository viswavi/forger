"""Computation graph visualization for Forger DataSpecs.

Decoupled from core/runner — this module only reads DataSpecs. It knows
how to recurse into FOREACH body subgraphs and COND branches so the
full computation (including per-row routing) shows up in the viz.
"""
from __future__ import annotations

from typing import Iterable

from pyforger.core import DataSpec, NodeType


_TYPE_COLORS = {
    NodeType.ACQUIRE:   "#B8E6B8",
    NodeType.ALTER:     "#B8D4E6",
    NodeType.FILTER:    "#E6C8B8",
    NodeType.DECOMPOSE: "#E6B8E6",
    NodeType.RAW:       "#DDDDDD",
    NodeType.ITEM:      "#FFE599",
    NodeType.EXPR:      "#E0E0E0",
    NodeType.FOREACH:   "#F4CCCC",
    NodeType.COND:      "#D9EAD3",
}


def _children(node: DataSpec) -> list[DataSpec]:
    """Return the node's effective children, dedup'd by identity."""
    kids: list[DataSpec] = list(node.inputs)
    if node.node_type == NodeType.FOREACH:
        body = getattr(node, "_foreach_body", None)
        if isinstance(body, DataSpec):
            kids.append(body)
    elif node.node_type == NodeType.COND:
        for attr in ("_cond_pred", "_cond_then", "_cond_else"):
            c = getattr(node, attr, None)
            if isinstance(c, DataSpec):
                kids.append(c)
    seen: set[int] = set()
    out: list[DataSpec] = []
    for k in kids:
        if id(k) not in seen:
            seen.add(id(k))
            out.append(k)
    return out


def _escape_mermaid(s: str) -> str:
    """Mermaid labels inside ["..."] don't decode HTML entities, so strip
    quotes and newlines rather than encode them."""
    return (
        s.replace('"', "'")
         .replace("\n", " ")
         .replace("\r", " ")
    )


def _escape_dot(s: str) -> str:
    return (
        s.replace('"', '\\"')
         .replace("\n", "\\n")
         .replace("\r", " ")
    )


def _edge_label(parent: DataSpec, child: DataSpec) -> str | None:
    if parent.node_type == NodeType.FOREACH and child is getattr(parent, "_foreach_body", None):
        return "body"
    if parent.node_type == NodeType.FOREACH and child is getattr(parent, "_foreach_source", None):
        return "source"
    if parent.node_type == NodeType.COND:
        if child is getattr(parent, "_cond_pred", None):  return "pred"
        if child is getattr(parent, "_cond_then", None):  return "then"
        if child is getattr(parent, "_cond_else", None):  return "else"
    return None


def _walk(root: DataSpec) -> list[DataSpec]:
    """All nodes reachable (including FOREACH body and COND branches) in topo order."""
    seen: dict[int, DataSpec] = {}
    order: list[DataSpec] = []

    def dfs(node: DataSpec) -> None:
        if id(node) in seen:
            return
        seen[id(node)] = node
        for dep in _children(node):
            dfs(dep)
        order.append(node)

    dfs(root)
    return order


def _label(node: DataSpec) -> str:
    status = node._status.name.lower()
    return f"{node.name} : {node.node_type.name} ({status})"


def print_tree(spec: DataSpec, *, _prefix: str = "", _is_last: bool = True,
               _edge: str | None = None, _seen: set[int] | None = None) -> None:
    """ASCII tree of the DAG, showing FOREACH bodies and COND branches."""
    if _seen is None:
        _seen = set()
    connector = "└── " if _is_last else "├── "
    tag = f"[{_edge}] " if _edge else ""
    nid = id(spec)
    if nid in _seen:
        print(f"{_prefix}{connector}{tag}{_label(spec)}  (↺)")
        return
    _seen.add(nid)
    print(f"{_prefix}{connector}{tag}{_label(spec)}")
    child_prefix = _prefix + ("    " if _is_last else "│   ")
    kids = _children(spec)
    for i, dep in enumerate(kids):
        edge = _edge_label(spec, dep)
        print_tree(dep, _prefix=child_prefix, _is_last=(i == len(kids) - 1),
                   _edge=edge, _seen=_seen)


def to_mermaid(spec: DataSpec) -> str:
    """Mermaid flowchart. Paste into mermaid.live or GitHub md."""
    nodes = _walk(spec)
    id_of = {id(n): f"n{i}" for i, n in enumerate(nodes)}

    lines = ["flowchart TD"]
    for n in nodes:
        nid = id_of[id(n)]
        label = _escape_mermaid(_label(n))
        lines.append(f'    {nid}["{label}"]')
        color = _TYPE_COLORS.get(n.node_type, "#EEEEEE")
        lines.append(
            f"    style {nid} fill:{color},stroke:#222,stroke-width:1px,color:#000"
        )
    for n in nodes:
        for dep in _children(n):
            edge = _edge_label(n, dep)
            arrow = f'-- "{_escape_mermaid(edge)}" -->' if edge else "-->"
            lines.append(f"    {id_of[id(dep)]} {arrow} {id_of[id(n)]}")
    return "\n".join(lines)


def to_dot(spec: DataSpec) -> str:
    """Graphviz DOT. Render with `dot -Tpng graph.dot -o graph.png`."""
    nodes = _walk(spec)
    id_of = {id(n): f"n{i}" for i, n in enumerate(nodes)}

    lines = [
        "digraph Forger {",
        '    rankdir="TB";',
        '    node [shape=box, style=filled, fontname="Helvetica"];',
    ]
    for n in nodes:
        nid = id_of[id(n)]
        label = _escape_dot(_label(n))
        color = _TYPE_COLORS.get(n.node_type, "#EEEEEE")
        lines.append(f'    {nid} [label="{label}", fillcolor="{color}"];')
    for n in nodes:
        for dep in _children(n):
            edge = _edge_label(n, dep)
            extra = f' [label="{_escape_dot(edge)}"]' if edge else ""
            lines.append(f"    {id_of[id(dep)]} -> {id_of[id(n)]}{extra};")
    lines.append("}")
    return "\n".join(lines)


def save_png(spec: DataSpec, path: str) -> str:
    try:
        import graphviz  # type: ignore
    except ImportError as e:
        raise ImportError(
            "save_png requires the 'graphviz' Python package. "
            "Install it with `pip install graphviz`."
        ) from e
    src = graphviz.Source(to_dot(spec))
    try:
        return src.render(filename=path, format="png", cleanup=True)
    except graphviz.backend.ExecutableNotFound as e:  # type: ignore[attr-defined]
        raise RuntimeError(
            "Graphviz system binary `dot` not found on PATH. "
            "Install it: `brew install graphviz` (macOS) or "
            "`apt-get install graphviz` (Debian/Ubuntu)."
        ) from e


def summary(spec: DataSpec) -> dict:
    nodes = _walk(spec)
    counts: dict[str, int] = {}
    for n in nodes:
        counts[n.node_type.name] = counts.get(n.node_type.name, 0) + 1
    return {"total": len(nodes), "by_type": counts}


__all__ = ["print_tree", "to_mermaid", "to_dot", "save_png", "summary"]
