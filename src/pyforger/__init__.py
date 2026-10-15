"""Forger: a framework for building dataset and reward creation pipelines.

Pipelines are plain Python functions. Ops (in ``pyforger.acquire``,
``pyforger.alter``, ``pyforger.filter``, ``pyforger.inspect``) take and
return ``list[dict]`` directly. Group ops into ``@pyforger.step`` functions and
call those from a ``@pyforger.pipeline`` function to get cached steps, schema and
yield checks, and printed reports. Caching and tracing of single functions is
opt-in via ``pyforger.op``.

``pyforger.pipeline`` is the ``@pipeline`` decorator; the module of the same name
still provides ``run`` and ``optimize`` (``from pyforger.pipeline import run``).
"""

__version__ = "0.0.3"

import pyforger.acquire as acquire
import pyforger.alter as alter
import pyforger.filter as filter  # noqa: A001
import pyforger.inspect as inspect
from pyforger.op import op
from pyforger.pipeline import optimize, run

# Imported after the line above on purpose: this rebinds the attribute
# ``pyforger.pipeline`` from the module to the @pipeline decorator.
from pyforger.steps import pipeline, step

__all__ = [
    "acquire",
    "alter",
    "filter",
    "inspect",
    "op",
    "optimize",
    "pipeline",
    "run",
    "step",
]
