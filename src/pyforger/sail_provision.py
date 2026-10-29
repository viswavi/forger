"""Bounded retries for confirmed Sail pre-creation admission rejections."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from typing import Any

from pyforger.session_budget import SessionBudget

_REFRESH = "Sailbox concurrency limits are refreshing; retry shortly"


def _refresh_rejection(error: BaseException) -> bool:
    from sail.errors import SailboxCreationError

    if not isinstance(error, SailboxCreationError):
        return False
    if error.status_code not in (None, 429) or str(error) != _REFRESH:
        return False
    # No stable machine-readable refresh code exists in Sail 0.14. Accept only
    # this verified pre-creation rejection, never advisory retryable/5xx alone.
    body = error.body
    return (
        body is None
        or body == _REFRESH
        or body
        in (
            {"message": _REFRESH},
            {"error": _REFRESH},
        )
    )


def create_sailbox(
    create: Callable[..., Any],
    *,
    budget: SessionBudget,
    setup_times: list[float] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[float, float], float] = random.uniform,
    **kwargs: Any,
) -> Any:
    """Create at most five times within the caller's unchanged budget/permit.

    Only successful setup can receive the existing SDK provisioning exemption;
    rejected calls and backoff always consume wall time. Transport uncertainty
    must propagate because a box may already exist without an accessible ID.
    """
    for attempt in range(5):
        remaining = budget.remaining_s()
        started = time.monotonic()
        evidence = None
        if kwargs.get("image") is not None:
            from pyforger._sail_image_provenance import observe

            kwargs["image"], evidence = observe(kwargs["image"])
        try:
            box = create(
                **kwargs,
                timeout=remaining,
                image_build_timeout=max(1, math.ceil(remaining)),
            )
        except Exception as error:
            if evidence is not None:
                from pyforger._sail_image_provenance import failure_image, probe

                if failure_image(error):
                    probe(error, evidence, started + remaining)
                    error.sail_image_provenance = evidence
                    error.add_note("Sail image provenance: " + str(evidence))
            if attempt == 4 or not _refresh_rejection(error):
                raise
            cap = min(10.0, 2.0 * 2**attempt)
            sleep(min(jitter(cap / 2, cap), budget.remaining_s()))
        else:
            if setup_times is not None:
                setup_times.append(round(budget.exempt_elapsed_s(started), 3))
            return box
