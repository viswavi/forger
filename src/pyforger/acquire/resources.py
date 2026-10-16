"""Bounded public HTTPS retrieval with validated, pinned destinations."""

from __future__ import annotations

import ipaddress
import re
import socket
import ssl
import time
from urllib.parse import urljoin, urlsplit

import urllib3

_MAX_DOWNLOAD = 10 * 1024 * 1024

__all__ = ["fetch_public"]


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
    if not addresses or any(not _public_address(entry[4][0]) for entry in addresses):
        raise ValueError("Resource URL resolves to a non-public address")
    return parts.hostname.lower(), addresses[0][4][0]


def fetch_public(url: str, max_bytes: int = _MAX_DOWNLOAD) -> tuple[str, bytes, str]:
    """Fetch a public HTTPS resource with bounded bytes and pinned DNS."""
    if type(max_bytes) is not int or not 0 < max_bytes <= 32 * 1024 * 1024:
        raise ValueError("Invalid download byte limit")
    deadline = time.monotonic() + 60
    for _ in range(4):
        if time.monotonic() >= deadline:
            raise TimeoutError("Resource download deadline exceeded")
        hostname, address = _public_https_address(url)
        parts = urlsplit(url)
        target = parts.path or "/"
        if parts.query:
            target += "?" + parts.query
        pool = urllib3.HTTPSConnectionPool(
            address,
            port=443,
            assert_hostname=hostname,
            server_hostname=hostname,
            ssl_context=ssl.create_default_context(),
        )
        try:
            response = pool.urlopen(
                "GET",
                target,
                headers={"Host": f"[{hostname}]" if ":" in hostname else hostname},
                timeout=urllib3.Timeout(connect=5, read=5),
                retries=False,
                redirect=False,
                preload_content=False,
            )
            try:
                if 300 <= response.status < 400:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("Resource redirect is missing a location")
                    url = urljoin(url, location)
                    continue
                if response.status >= 400:
                    raise ValueError(f"Resource returned HTTP {response.status}")
                data = bytearray()
                for chunk in response.stream(64 * 1024, decode_content=True):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Resource download deadline exceeded")
                    data.extend(chunk)
                    if len(data) > max_bytes:
                        raise ValueError("Resource exceeds the download limit")
                return (
                    url,
                    bytes(data),
                    response.headers.get("content-type", "").lower(),
                )
            finally:
                response.close()
        finally:
            pool.close()
    raise ValueError("Too many resource redirects")
