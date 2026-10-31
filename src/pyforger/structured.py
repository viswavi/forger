"""Bounded generation with caller-owned, three-valued validation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Generic, TypeVar

from pyforger.measurements import Measurement
from pyforger.utils import run_concurrently

Request = TypeVar("Request")
Output = TypeVar("Output")


@dataclass(frozen=True)
class GenerationAttempt(Generic[Request, Output]):
    """A produced output and its validation verdict for one request."""

    request: Request
    output: Output
    measurement: Measurement


@dataclass(frozen=True)
class ValidatedGeneration(Generic[Request, Output]):
    """Accepted output and every evaluated attempt, in execution order."""

    output: Output
    attempts: tuple[GenerationAttempt[Request, Output], ...]


class ValidationExhausted(RuntimeError, Generic[Request, Output]):
    """Rejected outputs retained after the generation budget is exhausted."""

    def __init__(
        self, attempts: tuple[GenerationAttempt[Request, Output], ...]
    ) -> None:
        """Retain the history and expose the final rejected attempt."""
        self.attempts = attempts
        self.failure = attempts[-1]
        self.output = self.failure.output
        self.measurement = self.failure.measurement
        super().__init__(
            f"Validation exhausted after {len(attempts)} attempts: "
            f"{self.measurement.reason}"
        )


class GenerationInconclusive(RuntimeError, Generic[Request, Output]):
    """An ungraded output stops generation without attempting repairs."""

    def __init__(
        self, attempts: tuple[GenerationAttempt[Request, Output], ...]
    ) -> None:
        """Retain all outputs and the final inconclusive measurement."""
        self.attempts = attempts
        self.failure = attempts[-1]
        self.output = self.failure.output
        self.measurement = self.failure.measurement
        super().__init__(f"Generation inconclusive: {self.measurement.reason}")


@dataclass(frozen=True)
class GenerationResult(Generic[Request, Output]):
    """Every generation outcome as a return value, never raised for flow control.

    ``status`` is ``"accepted"`` (a validated output is available),
    ``"exhausted"`` (the attempt budget ran out on rejected outputs), or
    ``"inconclusive"`` (an ungraded output stopped generation early).
    ``output`` is the accepted output, or the final evaluated output when
    generation did not accept. Pipelines that branch on the outcome (write
    reports, continue to later stages) use this form; ``generate_validated``
    remains the exception-raising form for callers that prefer it.
    """

    status: str
    output: Output
    attempts: tuple[GenerationAttempt[Request, Output], ...]

    @property
    def accepted(self) -> bool:
        """Whether a validated output is available."""
        return self.status == "accepted"

    @property
    def measurement(self) -> Measurement:
        """The final attempt's measurement."""
        return self.attempts[-1].measurement


def generate_result(
    request: Request,
    *,
    generate: Callable[[Request], Output],
    validate: Callable[[Output], Measurement],
    revise: (
        Callable[[Request, GenerationAttempt[Request, Output]], Request] | None
    ) = None,
    max_attempts: int = 3,
) -> GenerationResult[Request, Output]:
    """Run bounded validated generation and return the outcome.

    Accept True, retry False, and stop on None; callback errors propagate.
    Without ``revise``, rejected requests are generated again unchanged.
    Revision runs only when another attempt remains. Histories retain the
    caller's objects, so callbacks should not mutate previous requests/outputs.
    Unlike ``generate_validated``, exhaustion and inconclusive outcomes are
    returned as data rather than raised.
    """
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    attempts: list[GenerationAttempt[Request, Output]] = []
    for index in range(max_attempts):
        output = generate(request)
        measurement = validate(output)
        if not isinstance(measurement, Measurement):
            raise TypeError("validate must return a Measurement")
        attempt = GenerationAttempt(request, output, measurement)
        attempts.append(attempt)
        history = tuple(attempts)
        if measurement.passed is True:
            return GenerationResult("accepted", output, history)
        if measurement.passed is None:
            return GenerationResult("inconclusive", output, history)
        if index + 1 == max_attempts:
            return GenerationResult("exhausted", output, history)
        if revise is not None:
            request = revise(request, attempt)
    raise AssertionError("unreachable")


def generate_batch(
    requests: Sequence[Request],
    *,
    generate: Callable[[Request], Output],
    validate: Callable[[Output], Measurement],
    revise: (
        Callable[[Request, GenerationAttempt[Request, Output]], Request] | None
    ) = None,
    max_attempts: int = 3,
    concurrency: int | None = 1,
) -> list[GenerationResult[Request, Output]]:
    """Run bounded validated generation independently for every request.

    Each request reaches its own terminal ``accepted``, ``exhausted`` or
    ``inconclusive`` state exactly as :func:`generate_result` would produce
    alone; the returned results align with ``requests`` by position.  No
    request is filtered out, and callback exceptions propagate from the
    failing request instead of being scored as rejection.  ``concurrency``
    bounds simultaneous per-request generations (``None`` runs one worker
    per request); every request still performs its own bounded serial
    generate-validate-revise sequence.
    """
    if concurrency is not None and (type(concurrency) is not int or concurrency < 1):
        raise ValueError("concurrency must be a positive integer or None")

    def run(request: Request) -> GenerationResult[Request, Output]:
        return generate_result(
            request,
            generate=generate,
            validate=validate,
            revise=revise,
            max_attempts=max_attempts,
        )

    return run_concurrently(
        [partial(run, request) for request in requests],
        concurrency=concurrency,
    )


def generate_validated(
    request: Request,
    *,
    generate: Callable[[Request], Output],
    validate: Callable[[Output], Measurement],
    revise: (
        Callable[[Request, GenerationAttempt[Request, Output]], Request] | None
    ) = None,
    max_attempts: int = 3,
) -> ValidatedGeneration[Request, Output]:
    """Accept True, retry False, and stop on None; callback errors propagate.

    Without ``revise``, rejected requests are generated again unchanged.
    Revision runs only when another attempt remains. Histories retain the
    caller's objects, so callbacks should not mutate previous requests/outputs.
    """
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    attempts: list[GenerationAttempt[Request, Output]] = []
    for index in range(max_attempts):
        output = generate(request)
        measurement = validate(output)
        if not isinstance(measurement, Measurement):
            raise TypeError("validate must return a Measurement")
        attempt = GenerationAttempt(request, output, measurement)
        attempts.append(attempt)
        history = tuple(attempts)
        if measurement.passed is True:
            return ValidatedGeneration(output, history)
        if measurement.passed is None:
            raise GenerationInconclusive(history)
        if index + 1 == max_attempts:
            raise ValidationExhausted(history)
        if revise is not None:
            request = revise(request, attempt)
    raise AssertionError("unreachable")
