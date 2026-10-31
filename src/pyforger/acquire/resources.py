"""Public HTTPS retrieval with validated, pinned destinations and a deadline.

Payloads stream through an unlinked temporary file so the byte budget and the
shared deadline are enforced while data arrives. The public API still returns
``bytes`` because callers index, hash, and persist the payload in memory.
"""

from __future__ import annotations

import ipaddress
import logging
import math
import re
import socket
import ssl
import tempfile
import time
from urllib.parse import urljoin, urlsplit

import urllib3

logger = logging.getLogger(__name__)

__all__ = ["fetch_public"]

_CONNECT_TIMEOUT = 30.0
_READ_TIMEOUT = 120.0
_MAX_ATTEMPTS = 6  # One initial attempt plus up to five transient retries.
_MAX_REDIRECTS = 3
_RETRY_BACKOFF = (0.5, 1.0, 2.0, 4.0, 8.0)
_MAX_RETRY_DELAY = 30.0
_STREAM_CHUNK = 64 * 1024
_PROGRESS_INTERVAL_BYTES = 4 * 1024 * 1024
_PROGRESS_INTERVAL_SECONDS = 5.0
_RETRYABLE_STATUSES = frozenset({429})
_TRANSIENT_ERRORS = (urllib3.exceptions.HTTPError, OSError)


class _DeadlineExceeded(Exception):
    """Internal marker: the shared download deadline expired."""


class _Retryable(Exception):
    """Internal marker: a transient transport or HTTP failure worth retrying."""

    def __init__(
        self,
        error: BaseException,
        *,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(str(error))
        self.error = error
        self.status = status
        self.retry_after = retry_after


class _Redirect(Exception):
    """Internal marker: a 3xx response whose target must be validated."""

    def __init__(self, url: str) -> None:
        super().__init__(url)
        self.url = url


def _public_address(address: str) -> bool:
    """Reject local and transition-network addresses before resource access."""
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address) and (
        ip.ipv4_mapped or ip.sixtofour or ip.teredo
    ):
        return False
    return ip.is_global and not ip.is_multicast and not ip.is_reserved


def _public_https_address(url: str) -> tuple[str, str]:
    """Resolve and validate an HTTPS origin, returning its host and pinned IP."""
    if (
        not isinstance(url, str)
        or len(url) > 8192
        or re.search(r"[\s\\\x00-\x1f\x7f]", url)
    ):
        raise ValueError("Invalid public HTTPS URL")
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.port not in (None, 443)
    ):
        raise ValueError("Resource URLs must be public HTTPS URLs without credentials")
    addresses = socket.getaddrinfo(
        parts.hostname, parts.port or 443, type=socket.SOCK_STREAM
    )
    public_addresses: list[str] = []
    for entry in addresses:
        address = entry[4][0]
        if not isinstance(address, str) or not _public_address(address):
            raise ValueError("Resource URL resolves to a non-public address")
        public_addresses.append(address)
    if not public_addresses:
        raise ValueError("Resource URL resolves to a non-public address")
    return parts.hostname.lower(), public_addresses[0]


def _validate_limit(max_bytes: int | None) -> None:
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes <= 0):
        raise ValueError(
            "Invalid download byte limit: expected a positive integer or None"
        )


def _validate_deadline(deadline: float | None) -> None:
    if deadline is None:
        return
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError(
            "Invalid download deadline: expected a finite monotonic timestamp or None"
        )


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise _DeadlineExceeded()


def _timeout(deadline: float | None) -> urllib3.Timeout:
    """Build timeouts whose total is capped by the remaining shared deadline."""
    if deadline is None:
        return urllib3.Timeout(total=None, connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _DeadlineExceeded()
    return urllib3.Timeout(
        total=remaining, connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT
    )


def _sleep(seconds: float) -> None:
    """Pause between attempts; tests substitute this to avoid real delays."""
    time.sleep(seconds)


def _retry_after(response: urllib3.BaseHTTPResponse) -> float | None:
    """Parse a numeric ``Retry-After`` header, ignoring HTTP-date values."""
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value.strip())
    except (AttributeError, ValueError):
        return None
    if not math.isfinite(seconds) or seconds <= 0:
        return None
    return min(seconds, _MAX_RETRY_DELAY)


def _target(url: str) -> str:
    parts = urlsplit(url)
    target = parts.path or "/"
    if parts.query:
        target += "?" + parts.query
    return target


def _stream_to_disk(
    response: urllib3.BaseHTTPResponse,
    url: str,
    max_bytes: int | None,
    deadline: float | None,
    started: float,
    attempt: int,
) -> tuple[str, bytes, str]:
    """Stream a response into a temporary file, enforcing size and deadline."""
    total = 0
    chunks = 0
    next_progress_bytes = _PROGRESS_INTERVAL_BYTES
    next_progress_at = time.monotonic() + _PROGRESS_INTERVAL_SECONDS
    with tempfile.TemporaryFile() as spool:
        try:
            for chunk in response.stream(_STREAM_CHUNK, decode_content=True):
                _check_deadline(deadline)
                spool.write(chunk)
                total += len(chunk)
                chunks += 1
                if max_bytes is not None and total > max_bytes:
                    raise ValueError(
                        "Resource exceeds the download limit: received at least "
                        f"{total} bytes; limit {max_bytes} bytes"
                    )
                now = time.monotonic()
                if total >= next_progress_bytes or now >= next_progress_at:
                    logger.info(
                        "Resource fetch progress: attempt=%d elapsed=%.3fs bytes=%d "
                        "chunks=%d",
                        attempt,
                        now - started,
                        total,
                        chunks,
                    )
                    next_progress_bytes = total + _PROGRESS_INTERVAL_BYTES
                    next_progress_at = now + _PROGRESS_INTERVAL_SECONDS
        except urllib3.exceptions.SSLError:
            # A broken or invalid TLS session is a security failure, not a
            # transient one: surface it without retrying.
            raise
        except _TRANSIENT_ERRORS as exc:
            raise _Retryable(exc) from exc
        spool.seek(0)
        data = spool.read()
    content_type = response.headers.get("content-type", "").lower()
    logger.info(
        "Resource fetch complete: attempt=%d elapsed=%.3fs bytes=%d chunks=%d "
        "content_type=%s",
        attempt,
        time.monotonic() - started,
        len(data),
        chunks,
        content_type or "unknown",
    )
    return url, data, content_type


def _fetch_once(
    url: str,
    max_bytes: int | None,
    deadline: float | None,
    started: float,
    attempt: int,
) -> tuple[str, bytes, str]:
    """Perform one pinned HTTPS request, classifying transient failures."""
    try:
        hostname, address = _public_https_address(url)
    except OSError as exc:
        raise _Retryable(exc) from exc
    logger.info(
        "Resource fetch attempt %d: host=%s elapsed=%.3fs",
        attempt,
        hostname,
        time.monotonic() - started,
    )
    pool = urllib3.HTTPSConnectionPool(
        address,
        port=443,
        assert_hostname=hostname,
        server_hostname=hostname,
        ssl_context=ssl.create_default_context(),
    )
    try:
        try:
            response = pool.urlopen(
                "GET",
                _target(url),
                headers={"Host": f"[{hostname}]" if ":" in hostname else hostname},
                timeout=_timeout(deadline),
                retries=False,
                redirect=False,
                preload_content=False,
            )
        except urllib3.exceptions.SSLError:
            # Certificate and TLS validation failures are security errors, not
            # transient transport faults: surface them without retrying.
            raise
        except (urllib3.exceptions.HTTPError, OSError) as exc:
            raise _Retryable(exc) from exc
        try:
            if 300 <= response.status < 400:
                location = response.headers.get("location")
                if not location:
                    raise ValueError("Resource redirect is missing a location")
                raise _Redirect(urljoin(url, location))
            if response.status in _RETRYABLE_STATUSES or response.status >= 500:
                raise _Retryable(
                    urllib3.exceptions.HTTPError(
                        f"Resource returned HTTP {response.status}"
                    ),
                    status=response.status,
                    retry_after=_retry_after(response),
                )
            if response.status >= 400:
                raise ValueError(f"Resource returned HTTP {response.status}")
            return _stream_to_disk(response, url, max_bytes, deadline, started, attempt)
        finally:
            response.close()
    finally:
        pool.close()


def _wait_before_retry(
    retries: int, failure: _Retryable, deadline: float | None, started: float
) -> None:
    delay = _RETRY_BACKOFF[min(retries, len(_RETRY_BACKOFF)) - 1]
    if failure.retry_after is not None:
        delay = max(delay, failure.retry_after)
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _DeadlineExceeded()
        delay = min(delay, remaining)
    logger.warning(
        "Resource fetch retry %d/%d after %s: elapsed=%.3fs delay=%.2fs",
        retries,
        _MAX_ATTEMPTS - 1,
        type(failure.error).__name__,
        time.monotonic() - started,
        delay,
    )
    _sleep(delay)


def _exhausted(failure: _Retryable) -> BaseException:
    if failure.status is None:
        return failure.error
    return ValueError(
        f"Resource returned HTTP {failure.status} after {_MAX_ATTEMPTS} attempts"
    )


def _fetch_with_retries(
    url: str, max_bytes: int | None, deadline: float | None, started: float
) -> tuple[str, bytes, str]:
    """Run the initial attempt plus up to five transient retries."""
    retries = 0
    redirects = 0
    while True:
        _check_deadline(deadline)
        try:
            return _fetch_once(url, max_bytes, deadline, started, retries + 1)
        except _Retryable as failure:
            retries += 1
            if retries >= _MAX_ATTEMPTS:
                raise _exhausted(failure) from failure
            _wait_before_retry(retries, failure, deadline, started)
        except _Redirect as redirect:
            redirects += 1
            if redirects > _MAX_REDIRECTS:
                raise ValueError("Too many resource redirects") from redirect
            url = redirect.url


def fetch_public(
    url: str, max_bytes: int | None = None, deadline: float | None = None
) -> tuple[str, bytes, str]:
    """Fetch a public HTTPS resource with pinned DNS and an optional byte budget.

    Args:
        url: Public HTTPS URL to retrieve.
        max_bytes: Optional positive byte ceiling; ``None`` means unlimited.
        deadline: Optional absolute ``time.monotonic()`` value shared by every
            attempt and redirect. When omitted, the request is bounded only by the
            connect and read-inactivity timeouts.

    Returns:
        The final URL, the payload as bytes, and the lowercased content type.

    Raises:
        ValueError: Invalid input, a non-public destination, a non-retryable HTTP
            status, a redirect violation, or a byte-limit overrun.
        TimeoutError: The shared deadline expired.
    """
    _validate_limit(max_bytes)
    _validate_deadline(deadline)
    started = time.monotonic()
    try:
        return _fetch_with_retries(url, max_bytes, deadline, started)
    except _DeadlineExceeded as exc:
        raise TimeoutError("Resource download deadline exceeded") from exc
