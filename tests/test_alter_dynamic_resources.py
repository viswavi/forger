"""Tests for dynamic pipeline operations."""

import importlib.metadata
import json
import socket
import socketserver
import threading
from contextlib import contextmanager

import pytest

from pyforger.alter import _dynamic_git_proxy as proxy_module
from pyforger.alter import _dynamic_resources as resources


def test_existing_record_is_unchanged(monkeypatch):
    """A retrieved search result's URL must not trigger another download."""
    monkeypatch.setattr(
        resources, "_download", lambda url: pytest.fail("unexpected fetch")
    )
    item = {
        "text": "# Python Challenges\nIs Disarium Number?",
        "id": "https://programiz.pro/community-challenges/python",
        "title": "Python Coding Challenges",
        "url": "https://programiz.pro/community-challenges/python",
        "query": "Very hard programming challenges in Python",
    }
    assert resources.prepare_input(item) is item


def test_html_url_is_extracted(monkeypatch):
    """Test html url is extracted."""
    monkeypatch.setattr(
        resources,
        "_download",
        lambda url: (
            b"<html><style>hidden</style><body>Hello <b>world</b></body></html>",
            "text/html",
        ),
    )
    result = resources.prepare_input({"url": "https://example.org/page"})
    assert result["kind"] == "html"
    assert "Hello" in result["content"]
    assert "hidden" not in result["content"]


def test_pdf_url_is_parsed(monkeypatch):
    # A small valid PDF with an actual text drawing operation. A blank PDF
    # would check parsing but would not exercise PdfReader.extract_text().
    """Test pdf url is parsed."""
    pytest.importorskip("pypdf")
    message = b"BT /F1 12 Tf 20 100 Td (Hello from PDF) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length "
        + str(len(message)).encode()
        + b" >>\nstream\n"
        + message
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(b"xref\n0 6\n0000000000 65535 f \n")
    for offset in offsets:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n"
        + str(xref).encode()
        + b"\n%%EOF"
    )
    monkeypatch.setattr(
        resources,
        "_download",
        lambda url: (
            bytes(pdf),
            "application/pdf",
        ),
    )
    result = resources.prepare_input({"url": "https://example.org/file.pdf"})
    assert result["kind"] == "pdf"
    assert result["content"].strip() == "Hello from PDF"


def test_git_url_is_cloned_to_bounded_text_bundle(monkeypatch):
    """Test git url is cloned to bounded text bundle."""
    monkeypatch.setattr(resources, "_public_https", lambda url: "github.com")
    monkeypatch.setenv("LITELLM_API_KEY", "dummy")

    def fake_clone(url, root):
        """Create a small repository snapshot."""
        repo = root / "repo"
        repo.mkdir()
        (repo / "main.py").write_text("def parse(): return 42")
        (repo / "binary.bin").write_bytes(b"\0binary")
        (repo / ".git").mkdir()
        (repo / ".git" / "config").write_text("internal")

    monkeypatch.setattr(resources, "_clone_repo", fake_clone)
    result = resources.prepare_input({"url": "https://github.com/acme/repo.git"})
    assert result["kind"] == "git"
    assert result["files"] == {"main.py": "def parse(): return 42"}


def test_git_rejects_other_hosts(monkeypatch):
    """Test git rejects other hosts."""
    monkeypatch.setattr(resources, "_public_https", lambda url: "example.org")
    with pytest.raises(ValueError, match="GitHub or GitLab"):
        resources.prepare_input({"url": "https://example.org/repo.git"})


def test_uppercase_git_suffix_is_accepted(monkeypatch):
    """Treat uppercase .GIT suffix as a repository URL."""
    monkeypatch.setattr(resources, "_public_https", lambda url: "github.com")

    def fake_clone(url, root):
        """Create a tiny snapshot instead of cloning."""
        repo = root / "repo"
        repo.mkdir()
        (repo / "README.md").write_text("ok")

    monkeypatch.setattr(resources, "_clone_repo", fake_clone)
    result = resources.prepare_input({"url": "https://github.com/acme/repo.GIT"})
    assert result["files"] == {"README.md": "ok"}


def test_plain_text_url_preserves_code(monkeypatch):
    """Do not strip comparison operators from source served as text/plain."""
    monkeypatch.setattr(
        resources,
        "_download",
        lambda url: (b"if a<b and b>0:", "text/plain; charset=utf-8"),
    )
    result = resources.prepare_input({"url": "https://example.org/source.py"})
    assert result == {
        "kind": "text",
        "url": "https://example.org/source.py",
        "content": "if a<b and b>0:",
    }


def test_https_rejects_private_addresses(monkeypatch):
    """Test https rejects private addresses."""
    monkeypatch.setattr(
        resources.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("127.0.0.1", 443))],
    )
    with pytest.raises(ValueError, match="non-public"):
        resources._public_https("https://localhost/document.pdf")


def test_download_validates_redirect_target(monkeypatch):
    """Test download validates redirect target."""
    monkeypatch.setattr(
        resources,
        "_public_https_address",
        lambda url: (
            (_ for _ in ()).throw(ValueError("blocked"))
            if "internal" in url
            else ("example.org", "93.184.215.14")
        ),
    )

    class Response:
        status = 302
        headers = {"location": "http://internal/file"}

        def close(self):
            """Close the mocked response."""

    class Pool:
        def __init__(self, address, **kwargs):
            assert address == "93.184.215.14"
            assert kwargs["assert_hostname"] == "example.org"
            assert kwargs["server_hostname"] == "example.org"

        def close(self):
            """Close the mock pool."""

        def urlopen(self, *args, **kwargs):
            """Return a redirect for the pinned address."""
            assert kwargs["headers"]["Host"] == "example.org"
            return Response()

    monkeypatch.setattr("urllib3.HTTPSConnectionPool", Pool)
    with pytest.raises(ValueError, match="blocked"):
        resources._download("https://example.org/start")


def test_download_uses_validated_ip_without_reresolving(monkeypatch):
    """Pin the connection target while preserving the HTTPS hostname."""
    lookups = []

    def getaddrinfo(host, port, **kwargs):
        """Return a public address once, then a private rebinding answer."""
        lookups.append(host)
        address = "93.184.215.14" if len(lookups) == 1 else "127.0.0.1"
        return [(None, None, None, None, (address, port))]

    class Response:
        status = 200
        headers = {"content-type": "text/plain"}

        def stream(self, *args, **kwargs):
            """Return one response chunk."""
            yield b"hello"

        def close(self):
            """Close the mocked response."""

    class Pool:
        def __init__(self, address, **kwargs):
            assert address == "93.184.215.14"
            assert kwargs["assert_hostname"] == "example.org"
            assert kwargs["server_hostname"] == "example.org"

        def urlopen(self, method, target, **kwargs):
            """Return a response for the pinned address."""
            assert target == "/page?q=1"
            assert kwargs["headers"]["Host"] == "example.org"
            return Response()

        def close(self):
            """Close the mocked pool."""

    monkeypatch.setattr(resources.socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr("urllib3.HTTPSConnectionPool", Pool)
    assert resources._download("https://example.org/page?q=1") == (
        b"hello",
        "text/plain",
    )
    assert lookups == ["example.org"]


def test_git_proxy_rejects_other_hosts():
    """Do not let Git redirects tunnel to arbitrary hosts."""
    from pyforger.alter._dynamic_git_proxy import git_proxy

    try:
        with git_proxy() as proxy:
            port = int(proxy.rsplit(":", 1)[1])
            with socket.create_connection(("127.0.0.1", port)) as connection:
                connection.sendall(b"CONNECT internal.example:443 HTTP/1.1\r\n\r\n")
                assert b"403 Forbidden" in connection.recv(1024)
    except PermissionError:
        pytest.skip("Local socket binding is unavailable in this sandbox")


def test_clone_stops_at_disk_limit(tmp_path, monkeypatch):
    """Stop Git before a checkout fills unbounded temporary storage."""
    with (tmp_path / "large.bin").open("wb") as stream:
        stream.truncate(resources._MAX_GIT_DISK + 1)
    killed = []

    @contextmanager
    def fake_proxy():
        """Avoid binding a real local socket."""
        yield "http://127.0.0.1:12345"

    class Process:
        pid = 12345
        returncode = None

        def poll(self):
            """Report an active clone."""
            return self.returncode

        def wait(self, timeout):
            """Mark the clone reaped."""
            self.returncode = -9

        def kill(self):
            """Mark the clone killed on non-POSIX platforms."""
            killed.append(self.pid)

    def fake_popen(command, **kwargs):
        """Verify Git is forced through the bounded local proxy."""
        assert "http.proxy=http://127.0.0.1:12345" in command
        assert "http.followRedirects=false" in command
        assert "LITELLM_API_KEY" not in kwargs["env"]
        return Process()

    monkeypatch.setattr(resources, "git_proxy", fake_proxy)
    monkeypatch.setattr(resources.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(resources.os, "killpg", lambda pid, sig: killed.append(pid))
    with pytest.raises(ValueError, match="disk limit"):
        resources._clone_repo("https://github.com/acme/repo.git", tmp_path)
    assert killed == [12345]


def test_git_tool_lists_and_reads_prepared_files(monkeypatch):
    """Test git tool lists and reads prepared files."""
    pytest.importorskip("openhands.sdk")
    from pyforger.alter import _dynamic_tool as tool

    prepared = {
        "kind": "git",
        "url": "https://github.com/a/b.git",
        "files": {"README.md": "hello", "src/a.py": "x = 1"},
    }
    monkeypatch.setattr(tool, "prepare_input", lambda item: prepared)
    session = tool.PythonExecutor({"url": "https://github.com/a/b.git"}, 5)
    git = tool.GitExecutor(session)
    listing = git(tool.GitAction(operation="list"))
    assert json.loads(listing.result_text)["files"] == ["README.md", "src/a.py"]
    content = git(tool.GitAction(operation="read", path="src/a.py"))
    assert json.loads(content.result_text)["content"] == "x = 1"
    assert session.prepared_input() == prepared


def test_sdk_resolves_all_restricted_tools():
    """Test sdk resolves all restricted tools."""
    pytest.importorskip("openhands.sdk")
    from openhands.sdk.tool import Tool
    from openhands.sdk.tool.registry import resolve_tool

    from pyforger.alter import _dynamic_tool as tool

    session = tool.PythonExecutor({"text": "plain input"}, 5)
    session_id = tool.register_executor(session)
    try:
        for name in ("ForgerPythonTool", "ForgerResourceTool", "ForgerGitTool"):
            resolved = resolve_tool(
                Tool(name=name, params={"session_id": session_id}), None
            )
            assert len(resolved) == 1
            assert resolved[0].executor is not None
    finally:
        tool.unregister_executor(session_id)


def test_generated_program_receives_prepared_url(monkeypatch):
    """Test generated program receives prepared url."""
    version = importlib.metadata.version("RestrictedPython")
    if tuple(map(int, version.split(".")[:2])) < (8, 4):
        pytest.skip("Dynamic worker requires RestrictedPython >=8.4")
    from pyforger import local_agent

    try:
        local_agent._require_dynamic_platform()
    except NotImplementedError:
        pytest.skip("Dynamic worker requires Unix signal timers")

    prepared = {"kind": "html", "url": "https://example.org", "content": "hello"}
    monkeypatch.setattr(local_agent, "prepare_input", lambda item: prepared)
    code = 'def run(item):\n    return [item["content"]]'
    assert local_agent._execute(code, {"url": "https://example.org"}, 5) == ["hello"]


def test_public_fetch_rejects_invalid_limits_before_network():
    """Reject malformed limits without opening a connection."""
    for limit in (0, -1, True, 33 * 1024 * 1024):
        with pytest.raises(ValueError, match="byte limit"):
            resources.fetch_public("https://example.org/data", limit)


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "10.0.0.1", "224.0.0.1", "::ffff:8.8.8.8", "2002:0808:0808::1"],
)
def test_public_address_rejects_special_networks(address):
    """Transition and private addresses cannot be public resources."""
    assert not resources.public_address(address)


def test_public_fetch_is_shared_with_dynamic_download():
    """Acquisition and dynamic inspection share the same fetch implementation."""
    from pyforger.acquire.resources import fetch_public

    assert fetch_public is resources.fetch_public


@contextmanager
def local_git_tunnel(monkeypatch):
    """Route an allowed CONNECT destination to a real loopback echo server."""
    received = []
    destinations = []

    class Echo(socketserver.BaseRequestHandler):
        def handle(self):
            while data := self.request.recv(1024):
                received.append(data)
                self.request.sendall(data)

    real_connect = socket.create_connection
    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), Echo) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def destination(url):
            destinations.append(url)
            return "github.com", "93.184.216.34"

        def connect(address, *args, **kwargs):
            if address == ("93.184.216.34", 443):
                address = server.server_address
            return real_connect(address, *args, **kwargs)

        # Substitute only destination discovery/routing, not sockets or relay I/O.
        monkeypatch.setattr(resources, "_public_https_address", destination)
        monkeypatch.setattr(proxy_module.socket, "create_connection", connect)
        try:
            with proxy_module.git_proxy() as proxy:
                yield (
                    "127.0.0.1",
                    int(proxy.rsplit(":", 1)[1]),
                ), received, destinations
        finally:
            server.shutdown()
            thread.join(timeout=2)


def open_git_tunnel(address):
    """Complete CONNECT before sending tunnel data."""
    connection = socket.create_connection(address, timeout=3)
    connection.sendall(b"CONNECT github.com:443 HTTP/1.1\r\nHost: github.com\r\n\r\n")
    response = bytearray()
    while not response.endswith(b"\r\n\r\n"):
        chunk = connection.recv(1)
        assert chunk, "Proxy closed before completing CONNECT"
        response.extend(chunk)
    assert bytes(response) == b"HTTP/1.1 200 Connection Established\r\n\r\n"
    return connection


def test_git_proxy_forwards_real_socket_bytes(monkeypatch):
    """CONNECT relays exact bytes in both directions to the pinned destination."""
    with local_git_tunnel(monkeypatch) as (address, received, destinations):
        with open_git_tunnel(address) as connection:
            payload = b"git upload-pack\x00\xff\n"
            connection.sendall(payload)
            response = bytearray()
            while len(response) < len(payload):
                chunk = connection.recv(len(payload) - len(response))
                assert chunk
                response.extend(chunk)
            assert bytes(response) == payload
        assert b"".join(received) == payload
        assert destinations == ["https://github.com"]


@pytest.mark.parametrize("reconnect", [False, True])
def test_git_proxy_cuts_off_aggregate_transfer(monkeypatch, reconnect):
    """The budget counts both directions and is shared across CONNECT sessions."""
    monkeypatch.setattr(proxy_module, "_MAX_TRANSFER", 8)
    with local_git_tunnel(monkeypatch) as (address, received, _):
        with open_git_tunnel(address) as connection:
            connection.sendall(b"1234")
            response = bytearray()
            while len(response) < 4:
                chunk = connection.recv(4 - len(response))
                assert chunk
                response.extend(chunk)
            assert bytes(response) == b"1234"
            if not reconnect:
                connection.sendall(b"x")
                assert connection.recv(1) == b""
        if reconnect:
            with open_git_tunnel(address) as connection:
                connection.sendall(b"x")
                assert connection.recv(1) == b""
        assert b"".join(received) == b"1234"
