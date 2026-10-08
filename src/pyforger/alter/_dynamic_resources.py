"""Bounded, read-only URL materialization for dynamic pipelines."""

from __future__ import annotations

import io
import ipaddress
import json
import os
import signal
import socket
import ssl
import subprocess
import tempfile
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from pyforger.alter._dynamic_git_proxy import git_proxy

_MAX_DOWNLOAD = 10 * 1024 * 1024
_MAX_TEXT = 1024 * 1024
_MAX_FILES = 200
_MAX_FILE = 64 * 1024
_MAX_GIT_DISK = 64 * 1024 * 1024


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def _public_https_address(url: str) -> tuple[str, str]:
    """Resolve and validate an HTTPS origin, returning its host and pinned IP."""
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
    if not addresses or any(
        not ipaddress.ip_address(entry[4][0]).is_global for entry in addresses
    ):
        raise ValueError("Resource URL resolves to a non-public address")
    return parts.hostname.lower(), addresses[0][4][0]


def _public_https(url: str) -> str:
    """Validate an HTTPS origin and return its hostname."""
    return _public_https_address(url)[0]


def _download(url: str) -> tuple[bytes, str]:
    import urllib3

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
                    if len(data) > _MAX_DOWNLOAD:
                        raise ValueError("Resource exceeds the download limit")
                return bytes(data), response.headers.get("content-type", "").lower()
            finally:
                response.close()
        finally:
            pool.close()
    raise ValueError("Too many resource redirects")


def _tree_size(root: Path) -> int:
    """Count bytes written by a clone, including Git objects and checkout files."""
    total = 0
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            path = Path(directory) / filename
            if not path.is_symlink():
                try:
                    total += path.stat().st_size
                except FileNotFoundError:
                    pass
    return total


def _clone_repo(url: str, root: Path) -> None:
    """Clone through the pinned proxy with transfer, disk, and time limits."""
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(root),
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_LFS_SKIP_SMUDGE": "1",
    }
    with git_proxy() as proxy:
        command = [
            "git",
            "-c",
            "protocol.file.allow=never",
            "-c",
            "credential.helper=",
            "-c",
            f"http.proxy={proxy}",
            "-c",
            "http.followRedirects=false",
            "clone",
            "--depth=1",
            "--single-branch",
            "--no-tags",
            "--quiet",
            url,
            str(root / "repo"),
        ]
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + 60
        completed = False
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Git clone deadline exceeded")
                if _tree_size(root) > _MAX_GIT_DISK:
                    raise ValueError("Git clone exceeds the disk limit")
                time.sleep(0.05)
            if _tree_size(root) > _MAX_GIT_DISK:
                raise ValueError("Git clone exceeds the disk limit")
            if process.returncode:
                raise RuntimeError("Git clone failed")
            completed = True
        finally:
            if not completed:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                elif process.poll() is None:
                    process.kill()
                if process.poll() is None:
                    process.wait(timeout=5)


def _clone_files(url: str) -> dict[str, str]:
    host = _public_https(url)
    parts = urlsplit(url)
    if (
        host not in {"github.com", "gitlab.com"}
        or not parts.path.lower().endswith(".git")
        or parts.query
        or parts.fragment
    ):
        raise ValueError("Git inputs must be HTTPS .git URLs on GitHub or GitLab")
    with tempfile.TemporaryDirectory(prefix="forger-git-") as workspace:
        root = Path(workspace)
        _clone_repo(url, root)
        if not (root / "repo").is_dir():
            raise RuntimeError("Git clone failed")
        files: dict[str, str] = {}
        total = 0
        for directory, names, filenames in os.walk(root / "repo"):
            names[:] = sorted(name for name in names if name != ".git")
            for filename in sorted(filenames):
                if len(files) >= _MAX_FILES or total >= _MAX_TEXT:
                    break
                path = Path(directory) / filename
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size > _MAX_FILE
                ):
                    continue
                raw = path.read_bytes()
                if b"\0" in raw:
                    continue
                content = raw.decode("utf-8", errors="replace")
                content = content[: _MAX_TEXT - total]
                files[path.relative_to(root / "repo").as_posix()] = content
                total += len(content)
            if len(files) >= _MAX_FILES or total >= _MAX_TEXT:
                break
        return files


def prepare_input(item: str) -> str:
    """Return raw non-URLs or a JSON text bundle for supported HTTPS resources."""
    if not item.startswith("https://"):
        return item
    path = urlsplit(item).path.lower()
    if path.endswith(".git"):
        return json.dumps({"kind": "git", "url": item, "files": _clone_files(item)})
    data, content_type = _download(item)
    if path.endswith(".pdf") or "application/pdf" in content_type:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError("Install pyforger[dynamic] for PDF inputs") from exc
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages[:100])
        kind = "pdf"
    elif "text/html" in content_type or "application/xhtml+xml" in content_type:
        extractor = _TextExtractor()
        charset = content_type.partition("charset=")[2].split(";")[0].strip()
        extractor.feed(data.decode(charset or "utf-8", errors="replace"))
        text = " ".join(extractor.parts)
        kind = "html"
    else:
        charset = content_type.partition("charset=")[2].split(";")[0].strip()
        text = data.decode(charset or "utf-8", errors="replace")
        kind = "text"
    return json.dumps(
        {"kind": kind, "url": item, "content": text[:_MAX_TEXT]},
        ensure_ascii=False,
    )
