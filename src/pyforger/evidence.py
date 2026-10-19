"""Bounded public resource sessions and retained text evidence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pyforger.acquire.resources import fetch_public


class PublicEvidence:
    """Cache public HTTPS fetches and issue authoritative download receipts."""

    def __init__(
        self,
        output: Path,
        *,
        max_bytes: int = 32 * 1024 * 1024,
        max_total_bytes: int = 192 * 1024 * 1024,
        page_chars: int = 24000,
    ) -> None:
        """Configure bounded public evidence storage and retrieval."""
        if min(max_bytes, max_total_bytes, page_chars) < 1:
            raise ValueError("Evidence limits must be positive")
        self.output = Path(output)
        self.max_bytes, self.max_total_bytes = max_bytes, max_total_bytes
        self.page_chars = page_chars
        self.downloads: dict[str, dict[str, Any]] = {}
        self.pages: dict[str, tuple[str, bytes, str]] = {}
        self.total = 0

    def fetch(
        self,
        url: str,
        *,
        offset: int = 0,
        filename: str | None = None,
        license: str = "",
    ) -> str:
        """Return paged untrusted text or a receipt for an exclusive file write."""
        if offset < 0:
            raise ValueError("offset must be nonnegative")
        if filename:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", filename):
                raise ValueError("Filename must be a simple basename")
            if not license.strip():
                raise ValueError("Supply source license or 'unknown'")
            if filename in self.downloads:
                existing = self.downloads[filename]
                if existing["url"] != url or existing["license"] != license:
                    raise ValueError(
                        "Download filename already belongs to other evidence"
                    )
                return json.dumps(existing)
        if url not in self.pages:
            remaining = self.max_total_bytes - self.total
            if remaining <= 0:
                raise ValueError("Session byte budget exhausted")
            fetched = fetch_public(url, min(self.max_bytes, remaining))
            self.total += len(fetched[1])
            self.pages[url] = fetched
        final_url, body, content_type = self.pages[url]
        if filename:
            self.output.mkdir(parents=True, exist_ok=True)
            if self.output.is_symlink():
                raise ValueError("Evidence output cannot be a symlink")
            with (self.output / filename).open("xb") as stream:
                stream.write(body)
            receipt = dict(
                url=final_url,
                filename=filename,
                sha256=hashlib.sha256(body).hexdigest(),
                license=license,
            )
            self.downloads[filename] = receipt
            return json.dumps(receipt)
        if b"\x00" in body[:4096] or body.startswith((b"%PDF", b"PK\x03\x04")):
            return json.dumps(
                dict(
                    url=final_url,
                    content_type=content_type,
                    bytes=len(body),
                    note="Binary; download using filename.",
                )
            )
        text = body.decode("utf-8", errors="replace")
        end = offset + self.page_chars
        return json.dumps(
            dict(
                url=final_url,
                total_chars=len(text),
                offset=offset,
                next_offset=end if end < len(text) else None,
                untrusted_web_content=text[offset:end],
            )
        )
