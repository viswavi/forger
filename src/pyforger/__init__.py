"""Forger: a framework for building dataset and reward creation pipelines.

Pipelines are plain Python functions. Ops (in ``pyforger.acquire``,
``pyforger.alter``, ``pyforger.filter``, ``pyforger.inspect``) take and
return ``list[dict]`` directly. Caching, tracing, and pipeline-level optimization are
opt-in via ``pyforger.op`` / ``pyforger.pipeline``.
"""

__version__ = "0.0.3"

import pyforger.acquire as acquire
import pyforger.alter as alter
import pyforger.filter as filter  # noqa: A001
import pyforger.inspect as inspect
from pyforger.op import op
from pyforger.pipeline import optimize, run

__all__ = [
    "acquire",
    "alter",
    "filter",
    "inspect",
    "op",
    "optimize",
    "run",
]
