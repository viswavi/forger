"""Bounded generation with caller-owned, three-valued validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from pyforger.measurements import Measurement

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


def generate_validated(
    request: Request,
    *,
    generate: Callable[[Request], Output],
    validate: Callable[[Output], Measurement],
    revise: Callable[[Request, GenerationAttempt[Request, Output]], Request]
    | None = None,
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
