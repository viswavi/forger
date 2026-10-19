"""Row identity in step reports: which calls are chunks, and (python) rows.

A step's ``StepFlow`` sees the rows each top-level op took and returned, and
holds them until the step returns, so ``id()``s stay unique. It answers two
questions, from the same record:

- Which earlier calls a call's input rows came from (``OpCall.after``). Chunks
  of one op (``gen(rows[0:2])``, ``gen(rows[2:4])``) share a report row; a
  call fed by its own op's output (two judges in a row, A then B then A) gets
  its own row. ``group_calls`` makes the rows; the report uses it too.
- Which rows plain Python dropped or added between those report rows
  (``[r for r in rows if r["papers"]]``), shown as a dim ``(python)`` row,
  e.g. ``(python)  356 → 330  0.93``. By row identity when the next op's rows
  all come from the previous op's output (certain); else by row counts, when
  Python rebuilt the rows (``{**r, "x": 1}``) and each report row's calls are
  contiguous, marked inferred.

Anything else stays silent: rows merged from two sources, rows split between
two ops (also rebuilt ones), steps called with several lists, ops whose input
is not a list, and chunks of rebuilt rows. Between steps, only identity
counts, per stream when a step returned a dict of row lists. Past
``MAX_HELD_ROWS`` held rows a step stops tracking: later calls share report
rows by op name, and the step shows no (python) rows. Nothing here imports
the rest of pyforger.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

# A step stops tracking identity past this many held row references, so a
# huge step does not keep every row alive.
MAX_HELD_ROWS = 500_000


class PythonGap(BaseModel):
    """Rows that changed in plain Python between two points of a step or pipeline.

    In a step, ``after_call`` is the position in ``ops`` of the first call of
    the report row whose output went in (None: the step's input), and
    ``before_call`` that of the row that took the rows (None: the step's
    return value); ``after`` / ``before`` are their op names. In a pipeline,
    ``position`` is the index in ``steps`` of the step that took the rows,
    ``after`` / ``before`` are step names, and ``stream`` names the output
    stream of ``after`` the rows came from when it returned a dict of row
    lists. ``inferred`` is True when the rows were rebuilt and only counts
    compare.
    """

    after: str | None = None
    before: str | None = None
    after_call: int | None = None
    before_call: int | None = None
    position: int | None = None
    stream: str | None = None
    n_in: int
    n_out: int
    inferred: bool = False

    @property
    def yield_ratio(self) -> float | None:
        """``n_out / n_in``, or None without input rows."""
        return self.n_out / self.n_in if self.n_in else None


def traceable(row: Any) -> bool:
    """True when ``id(row)`` names one row: not a str, number, or None.

    Python shares equal strings and small numbers, so their ids say nothing.
    """
    return row is not None and not isinstance(
        row, (str, bytes, int, float, complex, bool, tuple, frozenset)
    )


def _ids(rows: Iterable[Any]) -> set[int]:
    return {id(r) for r in rows if traceable(r)}


def _head(op_name: str) -> str:
    return op_name.split(".", 1)[0]


def group_calls(
    names: list[str],
    afters: list[list[int]],
    labels: list[str | None] | None = None,
    new_rows: list[bool | None] | None = None,
) -> list[list[int]]:
    """Positions of a step's calls, one list per report row, by first call.

    A call joins the latest row of its op unless its input rows came from that
    row's output (``afters[i]``, see ``StepFlow``): chunks share a row, while
    two judges in a row, or a refine pass over generated rows, get one each.

    Rows rebuilt in Python (``{**r, ...}``) hide that chain, so two more
    checks keep calls apart: score labels that differ
    (``by_predicate(..., label="runs")`` then ``label="3 outputs"``), and
    input rows new to the step (``new_rows[i]``) after a row of calls that
    took known rows, or the other way round. None matches anything.
    """
    n = len(names)
    labels = labels if labels is not None else [None] * n
    new_rows = new_rows if new_rows is not None else [None] * n
    groups: list[list[int]] = []
    # Per group: its first label and new_rows flag that are not None.
    keys: list[list[Any]] = []
    latest: dict[str, int] = {}  # op name -> index in groups
    for i, (name, after) in enumerate(zip(names, afters)):
        g = latest.get(name)
        if (
            g is not None
            and not set(groups[g]) & set(after)
            and _matches(keys[g][0], labels[i])
            and _matches(keys[g][1], new_rows[i])
        ):
            groups[g].append(i)
            keys[g] = [
                old if old is not None else new
                for old, new in zip(keys[g], (labels[i], new_rows[i]))
            ]
            continue
        latest[name] = len(groups)
        groups.append([i])
        keys.append([labels[i], new_rows[i]])
    return groups


def _matches(a: Any, b: Any) -> bool:
    return a is None or b is None or a == b


@dataclass
class _Call:
    name: str
    rows_in: list[Any] | None  # None: an acquire or inspect op, or not a list
    rows_out: list[Any] | None  # None: an inspect op, an error, or not a list
    in_ids: set[int]
    out_ids: set[int]
    after: set[int]
    label: str | None = None  # the call's score label, e.g. "kept"
    new_rows: bool | None = None  # see OpCall.new_rows


@dataclass
class _Node:
    """The step input, one report row (its calls), or the step's return value."""

    name: str | None
    first: int | None = None  # position of the row's first call in ``ops``
    ins: list[Any] | None = None
    outs: list[Any] | None = None
    takes_rows: bool = True  # False for the step input and acquire ops
    raw_in_ids: set[int] = field(default_factory=set)  # also when ins is unknown
    in_ids: set[int] = field(default_factory=set)
    out_ids: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.in_ids = _ids(self.ins or [])
        self.out_ids = _ids(self.outs or [])
        self.raw_in_ids = self.raw_in_ids or set(self.in_ids)


class StepFlow:
    """The rows each top-level op of one step run took and returned.

    ``rows_in`` is the step's row list, or None when the step was called
    without exactly one list argument; ``arg_lists`` are all its list
    arguments. Call ``close`` when the step returns, to let the rows go.
    """

    def __init__(self, rows_in: list[Any] | None, arg_lists: list[list[Any]]):
        """Hold the step's input rows."""
        self._lock = threading.Lock()
        self.source = list(rows_in) if rows_in is not None else None
        self.arg_rows = [row for rows in arg_lists for row in rows]
        self.calls: list[_Call] = []
        # Ids of the step's input rows and every call's rows so far.
        self.seen = _ids(self.arg_rows)
        self.held = len(self.arg_rows)
        self.overflow = self.held > MAX_HELD_ROWS

    def add(self, ops: list[Any], call: Any, rows_in: Any, rows_out: Any) -> None:
        """Set ``call.after`` from row identity and append ``call`` to ``ops``.

        ``call.after`` lists the earlier calls whose output rows this call's
        input rows are, directly or through other ops. Rows copied in plain
        Python (``{**row}``) start a new lineage; ``call.new_rows`` says so.
        """
        head = _head(call.op_name)
        failed = getattr(call, "error", None) is not None
        listed_in = isinstance(rows_in, list)
        ins = (
            list(rows_in) if listed_in and head not in ("acquire", "inspect") else None
        )
        outs = (
            list(rows_out)
            if isinstance(rows_out, list) and head != "inspect" and not failed
            else None
        )
        in_ids = _ids(rows_in) if listed_in else set()
        with self._lock:
            if self.overflow or self.held + len(ins or []) + len(outs or []) > (
                MAX_HELD_ROWS
            ):
                self.overflow = True
                self.calls.clear()
                self.source, self.arg_rows = None, []
                self.seen = set()
                call.after = []
                call.new_rows = None
                ops.append(call)
                return
            self.held += len(ins or []) + len(outs or [])
            parents = {j for j, c in enumerate(self.calls) if c.out_ids & in_ids}
            after = set(parents)
            for j in parents:
                after |= self.calls[j].after
            call.after = sorted(after)
            if ins is not None and in_ids:
                call.new_rows = not in_ids & self.seen
            out_ids = _ids(outs or [])
            # Only held rows: an id of a row let go may name a new row later.
            self.seen |= (in_ids if ins is not None else set()) | out_ids
            scores = getattr(call, "scores", None)
            self.calls.append(
                _Call(
                    call.op_name,
                    ins,
                    outs,
                    in_ids,
                    out_ids,
                    after,
                    scores.label if scores is not None else None,
                    call.new_rows,
                )
            )
            ops.append(call)

    def close(self) -> None:
        """Drop every held row."""
        with self._lock:
            self.calls.clear()
            self.source = None
            self.arg_rows = []
            self.seen = set()

    def gaps(self, result: list[Any]) -> list[PythonGap]:
        """The (python) rows of this step, given the list it returned."""
        with self._lock:
            if self.overflow:
                return []
            calls = list(self.calls)
        groups = [
            g
            for g in group_calls(
                [c.name for c in calls],
                [sorted(c.after) for c in calls],
                [c.label for c in calls],
                [c.new_rows for c in calls],
            )
            if _head(calls[g[0]].name) != "inspect"  # reports, not rows
        ]
        # Counts compare only when each report row's calls are contiguous, so
        # the rows are in the order the rows flowed through them.
        order = sorted((i, n) for n, g in enumerate(groups) for i in g)
        runs = sum(1 for k, (_, n) in enumerate(order) if not k or order[k - 1][1] != n)
        contiguous = runs == len(groups)

        nodes = [_Node(None, outs=self.source, takes_rows=False)]
        known = _ids(self.arg_rows) | nodes[0].out_ids
        for g in groups:
            node = _group_node([calls[i] for i in g], g[0], known)
            known |= node.raw_in_ids | node.out_ids
            nodes.append(node)
        nodes.append(_Node(None, ins=list(result)))

        # later[i]: rows that nodes after i take, to spot rows split between ops.
        later: list[set[int]] = [set() for _ in nodes]
        for i in range(len(nodes) - 2, -1, -1):
            later[i] = later[i + 1] | nodes[i + 1].raw_in_ids

        # The stream: each node takes the rows of the latest node before it
        # (acquire ops take none, so they only pass rows on).
        preds: list[int] = [0] * len(nodes)
        prev = 0
        for i in range(1, len(nodes)):
            preds[i] = prev
            prev = i
        # Rows that first appear in a later node (rebuilt in Python) may be the
        # rows an earlier gap calls dropped. Unless they are simply the
        # previous node's rows rebuilt (it passed none on by identity), any
        # earlier gap is unsure.
        seen = _ids(self.arg_rows)
        unsure = [False] * len(nodes)
        for i, node in enumerate(nodes):
            fresh = node.raw_in_ids - seen
            if i and fresh:
                p = nodes[preds[i]]
                rebuilt = (
                    fresh == node.raw_in_ids
                    and p.outs is not None
                    and not p.out_ids & later[preds[i]]
                )
                unsure[i] = not rebuilt
            seen |= node.raw_in_ids | node.out_ids
        unsure_after = [False] * len(nodes)
        for i in range(len(nodes) - 2, -1, -1):
            unsure_after[i] = unsure_after[i + 1] or unsure[i + 1]

        out: list[PythonGap] = []
        earlier = _ids(self.arg_rows)  # rows seen before the current node
        for i, node in enumerate(nodes):
            if i and node.takes_rows and not unsure_after[i]:
                gap = _boundary(nodes[preds[i]], node, earlier, later[i], contiguous)
                if gap is not None:
                    out.append(gap)
            earlier |= node.raw_in_ids | node.out_ids
        return out


def _group_node(calls: list[_Call], first: int, known: set[int]) -> _Node:
    """One node for the calls of one report row.

    One call is a node as is. Several are chunks (``group_calls`` never puts
    a call with its own row's output): they join when every call takes rows
    seen before the row, and no row twice. Chunks of freshly rebuilt rows
    could be a chain through ``{**row}`` copies, so their rows stay unknown.
    """
    name = calls[0].name
    raw = set().union(*(c.in_ids for c in calls))
    outs = (
        None
        if any(c.rows_out is None for c in calls)
        else [r for c in calls for r in c.rows_out or []]
    )
    if _head(name) == "acquire":
        return _Node(name, first, outs=outs, takes_rows=False)
    if any(c.rows_in is None for c in calls):
        return _Node(
            name, first, outs=outs if len(calls) == 1 else None, raw_in_ids=raw
        )
    if len(calls) == 1:
        return _Node(name, first, ins=calls[0].rows_in, outs=outs)
    seen: set[int] = set()
    for c in calls:
        rows = c.rows_in or []
        if c.in_ids & seen or not c.in_ids <= known or len(c.in_ids) != len(rows):
            return _Node(name, first, raw_in_ids=raw)
        seen |= c.in_ids
    return _Node(
        name, first, ins=[r for c in calls for r in c.rows_in or []], outs=outs
    )


def _boundary(
    a: _Node, b: _Node, earlier: set[int], later: set[int], contiguous: bool
) -> PythonGap | None:
    """The gap between ``a``'s output and ``b``'s input, or None when unsure."""
    if a.outs is None or b.ins is None or (not a.outs and not b.ins):
        return None
    ins = b.ins
    if not all(traceable(r) for r in ins):
        return None
    in_ids = [id(r) for r in ins]
    # Rows from another op or the step input: Python merged two sources.
    if any(i not in a.out_ids and i in earlier for i in in_ids):
        return None
    # Some of a's rows went to a later op or the return value: a split.
    if (a.out_ids - set(in_ids)) & later:
        return None
    gap = PythonGap(
        after=a.name,
        before=b.name,
        after_call=a.first,
        before_call=b.first,
        n_in=len(a.outs),
        n_out=len(ins),
    )
    from_a = sum(1 for i in in_ids if i in a.out_ids)
    if ins and from_a == len(in_ids):
        return gap if gap.n_in != gap.n_out else None
    # Some of a's rows plus rebuilt ones: the rebuilt ones may be a's other
    # rows (a split), so counts say nothing.
    if from_a:
        return None
    # Rebuilt (or no) rows: only counts compare.
    if not contiguous or gap.n_in == gap.n_out:
        return None
    gap.inferred = True
    return gap


@dataclass
class _Candidate:
    gap: PythonGap
    after_index: int
    missing: set[int]


def _streams(rows_out: Any) -> dict[str | None, list[Any]] | None:
    """A step's output by stream: ``{None: rows}``, or a dict of row lists as is."""
    if isinstance(rows_out, list):
        return {None: list(rows_out)}
    if isinstance(rows_out, dict) and all(
        isinstance(v, list) for v in rows_out.values()
    ):
        return {str(k): list(v) for k, v in rows_out.items()}
    return None


class PipelineFlow:
    """Identity gaps between consecutive steps of one pipeline run.

    The output of the latest step is held until the next step starts (then it
    is compared and let go). A step that returned a dict of row lists is held
    per stream, so ``check(out["train"])`` compares with the ``train`` stream
    only. A drop is kept only when no later step, and not the pipeline's
    return value, takes the rows that went missing.
    """

    def __init__(self) -> None:
        """Start with no previous step."""
        self._lock = threading.Lock()
        self._prev: dict[str | None, list[Any]] | None = None
        self._prev_index = -1
        self._prev_name: str | None = None
        self._candidates: list[_Candidate] = []

    def start(
        self, rows_in: list[list[Any]], arg_lists: list[list[Any]]
    ) -> list[_Candidate]:
        """A step starts: compare its rows arguments with the previous step's output.

        For each stream of that output, the step's rows arguments drawn wholly
        from it (``out["train"]``, or ``solved[:4]`` and ``solved[4:]``) are
        joined; fewer rows than the stream had is a gap. A stream whose rows
        also went into another argument (merged with other rows) is skipped.
        """
        arg_ids = [_ids(rows) for rows in arg_lists]
        all_ids = set().union(*arg_ids)
        found: list[_Candidate] = []
        with self._lock:
            self._candidates = [c for c in self._candidates if not c.missing & all_ids]
            prev, self._prev = self._prev, None
            if prev is None:
                return found
            for stream, rows in prev.items():
                prev_ids = _ids(rows)
                taken = [
                    r
                    for r in rows_in
                    if r and all(traceable(x) for x in r) and _ids(r) <= prev_ids
                ]
                if not taken:
                    continue
                merged = any(
                    ids & prev_ids
                    for rows_arg, ids in zip(arg_lists, arg_ids)
                    if not any(rows_arg is t for t in taken)
                )
                taken_ids = set().union(*(_ids(r) for r in taken))
                n_out = sum(len(r) for r in taken)
                if merged or n_out >= len(rows):
                    continue
                candidate = _Candidate(
                    PythonGap(
                        after=self._prev_name,
                        stream=stream,
                        n_in=len(rows),
                        n_out=n_out,
                    ),
                    self._prev_index,
                    prev_ids - taken_ids,
                )
                self._candidates.append(candidate)
                found.append(candidate)
        return found

    def finish(
        self,
        candidates: list[_Candidate] | None,
        position: int,
        name: str,
        rows_out: Any,
    ) -> None:
        """A step joined the pipeline at ``position``; hold its output rows."""
        with self._lock:
            for candidate in candidates or []:
                candidate.gap.position = position
                candidate.gap.before = name
            self._prev = _streams(rows_out)
            self._prev_index, self._prev_name = position, name

    def close(
        self, result: Any, moved: dict[int, int] | None = None
    ) -> list[PythonGap]:
        """The pipeline returned ``result``: the gaps that stand, in step order.

        ``moved`` maps a step's position when it joined to its position in the
        summary (steps run with ``.submit`` join when they finish, but are
        listed in the order they were submitted).
        """
        returned = _returned_ids(result)

        def at(position: int) -> int:
            return moved.get(position, position) if moved else position

        with self._lock:
            gaps = []
            for c in self._candidates:
                if c.gap.position is None or c.missing & returned:
                    continue
                # Steps from worker threads may join out of order.
                if at(c.after_index) == at(c.gap.position) - 1:
                    c.gap.position = at(c.gap.position)
                    gaps.append(c.gap)
            self._candidates.clear()
            self._prev = None
        return sorted(gaps, key=lambda g: g.position or 0)


def _returned_ids(value: Any, depth: int = 3) -> set[int]:
    """Ids of the rows in a pipeline's return value.

    Rows may sit in containers: ``rows``, ``[kept, rest]``, ``(a, b)``, or
    ``{"train": rows, "test": rows}``, up to ``depth`` levels deep.
    """
    if isinstance(value, dict):
        items: Iterable[Any] = value.values()
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        return set()
    out: set[int] = set()
    for item in items:
        if traceable(item):
            out.add(id(item))  # a row, or a container of rows
        if depth > 1 and isinstance(item, (list, tuple, dict)):
            out |= _returned_ids(item, depth - 1)
    return out
