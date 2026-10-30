"""Progress for long model work: a tqdm bar in a terminal, else a log line every 30 s.

No heavy imports: data-parallel GPU workers import it.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from pyforger.utils import get_formatted_logger

LOG_INTERVAL_S = 30.0
_logger = get_formatted_logger("Progress")


def progress_mode(show_progress: bool | None) -> str:
    """Return "bar" in a terminal, "log" if stderr is redirected, or "off".

    `show_progress=None` reads `FORGER_PROGRESS` (on unless "0"/"false"/"no").
    """
    if show_progress is None:
        flag = os.environ.get("FORGER_PROGRESS", "").strip().lower()
        show_progress = flag not in {"0", "false", "no"}
    if not show_progress:
        return "off"
    isatty = getattr(sys.stderr, "isatty", None)
    return "bar" if isatty is not None and isatty() else "log"


def duration(seconds: float) -> str:
    """Short duration, e.g. "41s", "9m 12s", "10m", "1h 5m"."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s" if s % 60 else f"{s // 60}m"
    return f"{s // 3600}h {s % 3600 // 60}m"


def done_line(done: int, total: int, elapsed: float) -> str:
    """E.g. "120/500 done, 1m 31s, ~4m 48s left"."""
    left = ""
    if 0 < done < total:
        left = f", ~{duration(elapsed * (total - done) / done)} left"
    return f"{done}/{total} done, {duration(elapsed)}{left}"


class Progress:
    """Counts one job's finished units from any thread and shows them.

    ``"log"`` logs ``"<desc>: 120/500 done, 1m 31s, ~4m left"`` every 30 s (and
    once at the end if it logged); ``"bar"`` draws a tqdm bar; ``"off"`` is silent.
    """

    def __init__(self, desc: str, total: int, mode: str) -> None:
        """Start the clock; nothing shows until the first update."""
        self.desc, self.total, self.mode = desc, total, mode
        self.done = 0
        self._start = self._last = time.monotonic()
        self._logged = False
        self._lock = threading.Lock()
        self._bar = None
        if mode == "bar":
            from tqdm import tqdm

            self._bar = tqdm(total=total, desc=desc, leave=False, dynamic_ncols=True)

    def update(self, n: int = 1) -> None:
        """Count `n` more finished units."""
        with self._lock:
            self.done += n
            if self._bar is not None:
                self._bar.update(n)
            elif self.mode == "log" and time.monotonic() - self._last >= LOG_INTERVAL_S:
                self._log()

    def _log(self) -> None:
        now = time.monotonic()
        self._last, self._logged = now, True
        line = done_line(min(self.done, self.total), self.total, now - self._start)
        _logger.info("%s: %s", self.desc, line)

    def close(self) -> None:
        """Finish: the job ran to the end, so every unit counts as done."""
        with self._lock:
            self.done = self.total
            if self._bar is not None:
                self._bar.close()
            elif self._logged:
                self._log()

    def __enter__(self) -> Progress:
        """Use as ``with Progress(...) as p``."""
        return self

    def __exit__(self, *exc: Any) -> None:
        """Close the bar or log the last line."""
        self.close()


class _VllmBar:
    """The part of tqdm that vLLM's ``LLM.generate`` uses; counts go to ``sink``."""

    def __init__(
        self, sink: Callable[[int], None], iterable: Any = None, **kwargs: Any
    ) -> None:
        """Take tqdm's arguments from vLLM (total, desc, ...); only counts matter.

        vLLM also wraps its "Adding requests" loop; that iterable passes through.
        """
        self.sink = sink
        self.iterable = iterable
        self.n = 0
        self.total = kwargs.get("total")
        self.postfix: Any = ""
        self._start = time.monotonic()

    @property
    def format_dict(self) -> dict[str, Any]:
        """Give vLLM the ``elapsed`` time it divides tokens by."""
        return {"elapsed": max(time.monotonic() - self._start, 1e-6), "n": self.n}

    def __iter__(self) -> Any:
        """Iterate the wrapped iterable, uncounted."""
        return iter(() if self.iterable is None else self.iterable)

    def update(self, n: int = 1) -> None:
        """Forward ``n`` finished requests (or completions) to the sink."""
        self.n += n
        self.sink(n)

    def __getattr__(self, name: str) -> Any:
        # refresh, close, set_postfix, ...: nothing to draw.
        return lambda *args, **kwargs: None


def vllm_tqdm(sink: Callable[[int], None]) -> Callable[..., _VllmBar]:
    """A ``use_tqdm=`` value for vLLM that sends its progress counts to ``sink``."""
    return lambda iterable=None, **kwargs: _VllmBar(sink, iterable, **kwargs)
