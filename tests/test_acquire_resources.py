"""Transport tests for the public HTTPS fetcher in ``pyforger.acquire``.

Most tests drive a real TLS server on loopback: real sockets, a real X.509
certificate, real hostname verification, real streaming, and real retries. Only
two boundaries are substituted because they cannot exist locally -- DNS
resolution (the fetcher always dials port 443, and the SSRF guard is exercised
separately against real loopback addresses) and the trust anchor used by the
default TLS context.
"""

from __future__ import annotations

import datetime
import logging
import socket
import ssl
import tempfile
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import urllib3

from pyforger.acquire import resources as acquisition

REAL_POOL = urllib3.HTTPSConnectionPool
HOSTNAME = "example.org"
PINNED_IP = "127.0.0.1"


class _Route:
    """One canned response: status, headers, body, and optional pacing."""

    def __init__(
        self,
        status: int = 200,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
        chunk_size: int = 0,
        chunk_delay: float = 0.0,
        reset: bool = False,
    ) -> None:
        self.status = status
        self.body = body
        self.headers = headers or {}
        self.chunk_size = chunk_size
        self.chunk_delay = chunk_delay
        self.reset = reset


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: object) -> None:
        """Silence the default stderr request logging."""

    def do_GET(self) -> None:
        """Serve the next queued route for this path and record the request."""
        server = self.server
        server.requests.append((self.path, self.headers.get("Host")))
        route = server.next_route(self.path.partition("?")[0])
        if route.reset:
            self.close_connection = True
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            return
        self.send_response(route.status)
        for name, value in route.headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(route.body)))
        self.end_headers()
        if not route.body:
            return
        step = route.chunk_size or len(route.body)
        try:
            for offset in range(0, len(route.body), step):
                self.wfile.write(route.body[offset : offset + step])
                self.wfile.flush()
                if route.chunk_delay:
                    time.sleep(route.chunk_delay)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


class _Server(ThreadingHTTPServer):
    """Threaded HTTPS server with a per-path queue of canned responses."""

    daemon_threads = True

    def __init__(
        self,
        *args: object,
        routes: dict[str, list[_Route]],
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.routes = routes
        self.requests: list[tuple[str, str | None]] = []

    def next_route(self, path: str) -> _Route:
        """Pop the next response for ``path``; the last one repeats forever."""
        queue = self.routes.setdefault(path, [_Route(status=404)])
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def handle_error(self, request, client_address) -> None:
        """Drop noisy tracebacks from deliberately reset client sockets."""


def _self_signed(hostname: str, directory):
    """Write a self-signed certificate/key pair for ``hostname``."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(hostname)]), critical=False
        )
        .sign(key, hashes.SHA256())
    )
    cert_path = directory / "cert.pem"
    key_path = directory / "key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


@contextmanager
def tls_server(monkeypatch, tmp_path, routes, cert_hostname: str = HOSTNAME):
    """Serve ``routes`` over real TLS on loopback and pin the fetcher to it."""
    pytest.importorskip("cryptography")
    cert_path, key_path = _self_signed(cert_hostname, tmp_path)
    server = _Server(("127.0.0.1", 0), _Handler, routes=routes)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    server.socket = context.wrap_socket(server.socket, server_side=True)
    port = server.server_address[1]
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()

    def pool_factory(address, **kwargs):
        assert address == PINNED_IP
        # The transport always pins port 443; the fixture server listens
        # elsewhere, so swap in the ephemeral loopback port.
        assert kwargs.pop("port", 443) == 443
        return REAL_POOL(PINNED_IP, port=port, **kwargs)

    real_default_context = ssl.create_default_context

    def default_context(*args, **kwargs):
        trusted = real_default_context(*args, **kwargs)
        trusted.load_verify_locations(cafile=str(cert_path))
        return trusted

    monkeypatch.setattr(
        acquisition, "_public_https_address", lambda url: (HOSTNAME, PINNED_IP)
    )
    monkeypatch.setattr(acquisition.urllib3, "HTTPSConnectionPool", pool_factory)
    # Only the trust anchor is substituted; hostname verification stays enabled.
    monkeypatch.setattr(acquisition.ssl, "create_default_context", default_context)
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def record_sleeps(monkeypatch):
    """Capture retry backoff delays instead of waiting for them."""
    delays: list[float] = []
    monkeypatch.setattr(acquisition, "_sleep", delays.append)
    yield delays


@contextmanager
def fake_pool(monkeypatch, pool_class):
    """Replace the connection pool while keeping the rest of the fetch real."""
    monkeypatch.setattr(
        acquisition, "_public_https_address", lambda url: (HOSTNAME, "8.8.8.8")
    )
    monkeypatch.setattr(acquisition.urllib3, "HTTPSConnectionPool", pool_class)
    yield


def test_fetch_streams_through_temporary_disk_file(monkeypatch, tmp_path):
    """The payload passes through a real temp file before being returned."""
    body = bytes(range(256)) * 4096
    writes: list[int] = []
    real_temporary_file = tempfile.TemporaryFile

    class Spool:
        def __init__(self, inner):
            self._inner = inner

        def write(self, data):
            writes.append(len(data))
            return self._inner.write(data)

        def seek(self, *args):
            return self._inner.seek(*args)

        def read(self):
            return self._inner.read()

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *args):
            return self._inner.__exit__(*args)

    monkeypatch.setattr(
        acquisition.tempfile,
        "TemporaryFile",
        lambda *args, **kwargs: Spool(real_temporary_file(*args, **kwargs)),
    )
    with tls_server(monkeypatch, tmp_path, {"/data": [_Route(body=body)]}) as server:
        url, data, content_type = acquisition.fetch_public(f"https://{HOSTNAME}/data")
    assert data == body
    assert content_type == ""
    assert url == f"https://{HOSTNAME}/data"
    assert sum(writes) == len(body)
    assert len(writes) > 1, "body must arrive in streamed chunks, not one blob"
    assert [path for path, _ in server.requests] == ["/data"]


def test_fetch_pins_dns_and_sends_host_header(monkeypatch, tmp_path):
    """The pinned address carries the request while the Host header keeps TLS."""
    routes = {"/page": [_Route(body=b"hello")]}
    with tls_server(monkeypatch, tmp_path, routes) as server:
        _, data, _ = acquisition.fetch_public(f"https://{HOSTNAME}/page?q=1")
    assert data == b"hello"
    assert server.requests == [("/page?q=1", HOSTNAME)]


def test_fetch_retries_transient_statuses_then_succeeds(monkeypatch, tmp_path):
    """Repeated 503 responses are retried with backoff until one succeeds."""
    routes = {
        "/flaky": [
            _Route(status=503, body=b"unavailable"),
            _Route(status=503, body=b"unavailable"),
            _Route(body=b"recovered"),
        ]
    }
    with tls_server(monkeypatch, tmp_path, routes) as server:
        with record_sleeps(monkeypatch) as delays:
            _, data, _ = acquisition.fetch_public(f"https://{HOSTNAME}/flaky")
    assert data == b"recovered"
    assert delays == [0.5, 1.0]
    assert len(server.requests) == 3


def test_fetch_honors_retry_after(monkeypatch, tmp_path):
    """A 429 with Retry-After waits at least as long as the server asked."""
    routes = {
        "/limited": [
            _Route(status=429, body=b"slow", headers={"Retry-After": "7"}),
            _Route(body=b"ok"),
        ]
    }
    with tls_server(monkeypatch, tmp_path, routes):
        with record_sleeps(monkeypatch) as delays:
            _, data, _ = acquisition.fetch_public(f"https://{HOSTNAME}/limited")
    assert data == b"ok"
    assert delays == [7.0]


def test_fetch_gives_up_after_five_transient_retries(monkeypatch, tmp_path):
    """Six total attempts are made, then the HTTP status is reported."""
    routes = {"/down": [_Route(status=500)]}
    with tls_server(monkeypatch, tmp_path, routes) as server:
        with record_sleeps(monkeypatch) as delays:
            with pytest.raises(ValueError, match="HTTP 500 after 6 attempts"):
                acquisition.fetch_public(f"https://{HOSTNAME}/down")
    assert len(server.requests) == 6
    assert delays == [0.5, 1.0, 2.0, 4.0, 8.0]


def test_fetch_does_not_retry_client_errors(monkeypatch, tmp_path):
    """A non-retryable 4xx status fails on the first attempt."""
    routes = {"/gone": [_Route(status=404)]}
    with tls_server(monkeypatch, tmp_path, routes) as server:
        with pytest.raises(ValueError, match="HTTP 404"):
            acquisition.fetch_public(f"https://{HOSTNAME}/gone")
    assert len(server.requests) == 1


def test_fetch_does_not_retry_size_overruns(monkeypatch, tmp_path):
    """Exceeding the byte budget aborts immediately, without another attempt."""
    routes = {"/big": [_Route(body=b"x" * 4096)]}
    with tls_server(monkeypatch, tmp_path, routes) as server:
        with pytest.raises(ValueError, match="limit 1024 bytes"):
            acquisition.fetch_public(f"https://{HOSTNAME}/big", 1024)
    assert len(server.requests) == 1


def test_fetch_does_not_retry_tls_verification_failure(monkeypatch, tmp_path):
    """A certificate hostname mismatch is fatal and never retried."""
    routes = {"/secure": [_Route(body=b"secret")]}
    with tls_server(
        monkeypatch, tmp_path, routes, cert_hostname="wrong.example"
    ) as server:
        with record_sleeps(monkeypatch) as delays:
            with pytest.raises(urllib3.exceptions.SSLError):
                acquisition.fetch_public(f"https://{HOSTNAME}/secure")
    assert delays == []
    assert server.requests == []


def test_fetch_retries_transport_resets_then_succeeds(monkeypatch, tmp_path):
    """A dropped connection is retried, and the next attempt succeeds."""
    routes = {
        "/reset": [
            _Route(reset=True),
            _Route(body=b"second try"),
        ]
    }
    with tls_server(monkeypatch, tmp_path, routes) as server:
        with record_sleeps(monkeypatch) as delays:
            _, data, _ = acquisition.fetch_public(f"https://{HOSTNAME}/reset")
    assert data == b"second try"
    assert delays == [0.5]
    assert len(server.requests) == 2


def test_fetch_surfaces_transport_error_after_exhausting_retries(monkeypatch):
    """A persistent transport failure ends as the underlying error type."""
    attempts = []

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def urlopen(self, *args, **kwargs):
            attempts.append(1)
            raise urllib3.exceptions.ProtocolError("connection aborted")

        def close(self):
            pass

    with fake_pool(monkeypatch, Pool):
        with record_sleeps(monkeypatch):
            with pytest.raises(urllib3.exceptions.ProtocolError):
                acquisition.fetch_public(f"https://{HOSTNAME}/data")
    assert len(attempts) == 6


def test_fetch_follows_real_redirect_and_returns_final_url(monkeypatch, tmp_path):
    """A 302 is followed to the validated target and its URL is returned."""
    routes = {
        "/start": [_Route(status=302, headers={"Location": "/final"})],
        "/final": [_Route(body=b"final body")],
    }
    with tls_server(monkeypatch, tmp_path, routes) as server:
        url, data, _ = acquisition.fetch_public(f"https://{HOSTNAME}/start")
    assert data == b"final body"
    assert url == f"https://{HOSTNAME}/final"
    assert [path for path, _ in server.requests] == ["/start", "/final"]


def test_fetch_rejects_redirect_to_non_public_host(monkeypatch, tmp_path):
    """A redirect target is re-validated before any connection is attempted."""
    redirect = _Route(status=302, headers={"Location": "https://internal/x"})
    routes = {"/start": [redirect]}

    def validate(url: str) -> tuple[str, str]:
        if "internal" in url:
            raise ValueError("blocked")
        return HOSTNAME, PINNED_IP

    with tls_server(monkeypatch, tmp_path, routes) as server:
        monkeypatch.setattr(acquisition, "_public_https_address", validate)
        with pytest.raises(ValueError, match="blocked"):
            acquisition.fetch_public(f"https://{HOSTNAME}/start")
    assert [path for path, _ in server.requests] == ["/start"]


def test_fetch_bounds_redirect_chains(monkeypatch, tmp_path):
    """Redirect loops stop after the documented bound."""
    routes = {"/loop": [_Route(status=302, headers={"Location": "/loop"})]}
    with tls_server(monkeypatch, tmp_path, routes) as server:
        with pytest.raises(ValueError, match="Too many resource redirects"):
            acquisition.fetch_public(f"https://{HOSTNAME}/loop")
    assert len(server.requests) == 4


def test_fetch_deadline_bounds_a_stalled_body(monkeypatch, tmp_path):
    """A stalled transfer stops near the shared deadline, not the read timeout."""
    slow = _Route(body=b"x" * (1 << 20), chunk_size=4096, chunk_delay=1.0)
    routes = {"/slow": [slow]}
    with tls_server(monkeypatch, tmp_path, routes):
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="deadline exceeded"):
            acquisition.fetch_public(
                f"https://{HOSTNAME}/slow", deadline=time.monotonic() + 0.6
            )
        elapsed = time.monotonic() - started
    assert elapsed < 5.0


def test_fetch_deadline_is_shared_across_retries(monkeypatch, tmp_path):
    """Retry backoff cannot push work past the caller's absolute deadline."""
    routes = {"/down": [_Route(status=500)]}
    with tls_server(monkeypatch, tmp_path, routes) as server:
        started = time.monotonic()
        # Real backoff applies here: the first retry delay is clamped to the
        # remaining deadline, so the next loop check must report a timeout
        # instead of starting another attempt.
        with pytest.raises(TimeoutError, match="deadline exceeded"):
            acquisition.fetch_public(
                f"https://{HOSTNAME}/down", deadline=time.monotonic() + 0.4
            )
        elapsed = time.monotonic() - started
    assert elapsed < 3.0
    assert len(server.requests) == 1


def test_fetch_deadline_in_the_past_fails_before_connecting(monkeypatch, tmp_path):
    """An expired deadline is reported without opening a connection."""
    routes = {"/x": [_Route(body=b"x")]}
    with tls_server(monkeypatch, tmp_path, routes) as server:
        with pytest.raises(TimeoutError, match="deadline exceeded"):
            acquisition.fetch_public(
                f"https://{HOSTNAME}/x", deadline=time.monotonic() - 1
            )
    assert server.requests == []


@pytest.mark.parametrize("deadline", ["soon", True, float("nan"), float("inf")])
def test_fetch_rejects_invalid_deadlines(deadline):
    """Deadlines must be finite monotonic timestamps or None."""
    with pytest.raises(ValueError, match="Invalid download deadline"):
        acquisition.fetch_public("https://example.org/data", deadline=deadline)


def test_fetch_logs_timestamped_progress_and_totals(monkeypatch, tmp_path, caplog):
    """Progress records carry elapsed time, byte counts, and chunk counts."""
    body = b"y" * (10 << 20)
    routes = {"/large": [_Route(body=body, chunk_size=1 << 16)]}
    with tls_server(monkeypatch, tmp_path, routes):
        with caplog.at_level(logging.INFO, logger="pyforger.acquire.resources"):
            _, data, _ = acquisition.fetch_public(f"https://{HOSTNAME}/large")
    assert len(data) == len(body)
    messages = [record.getMessage() for record in caplog.records]
    progress = [m for m in messages if "Resource fetch progress" in m]
    complete = [m for m in messages if "Resource fetch complete" in m]
    assert progress, messages
    assert all("elapsed=" in m and "bytes=" in m and "chunks=" in m for m in progress)
    assert any(f"bytes={len(body)}" in m for m in complete)
    assert any("chunks=" in m for m in complete)


def test_fetch_uses_inactivity_timeouts_without_a_total(monkeypatch):
    """No deadline means connect/read bounds only, never a fixed total."""
    seen = {}

    class Response:
        status = 200
        headers = {"content-type": "text/plain"}

        def stream(self, size, *, decode_content):
            yield b"ok"

        def close(self):
            pass

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def urlopen(self, method, target, **kwargs):
            seen["timeout"] = kwargs["timeout"]
            return Response()

        def close(self):
            pass

    with fake_pool(monkeypatch, Pool):
        assert acquisition.fetch_public(f"https://{HOSTNAME}/data")[1] == b"ok"
    timeout = seen["timeout"]
    assert timeout.total is None
    assert timeout.connect_timeout == 30
    assert timeout.read_timeout == 120


def test_fetch_caps_total_timeout_at_the_remaining_deadline(monkeypatch):
    """A caller deadline shortens the per-attempt total timeout."""
    seen = {}

    class Response:
        status = 200
        headers = {"content-type": "text/plain"}

        def stream(self, size, *, decode_content):
            yield b"ok"

        def close(self):
            pass

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def urlopen(self, method, target, **kwargs):
            seen["timeout"] = kwargs["timeout"]
            return Response()

        def close(self):
            pass

    deadline = time.monotonic() + 2.0
    with fake_pool(monkeypatch, Pool):
        body = acquisition.fetch_public(f"https://{HOSTNAME}/data", deadline=deadline)[
            1
        ]
    assert body == b"ok"
    timeout = seen["timeout"]
    # The total is the remaining deadline; read inactivity still gets its own
    # bound, so a slow-but-progressing body is not cut short by the read limit.
    assert timeout.total is not None
    assert 0 < timeout.total <= 2.0
    assert timeout.read_timeout == 120


def test_fetch_rejects_malformed_limits_before_connecting():
    """Malformed byte limits fail without any transport activity."""
    for limit in (0, -1, True, 1.5, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="byte limit"):
            acquisition.fetch_public("https://example.org/data", limit)


@pytest.mark.parametrize("url", ["http://example.org/x", "https://127.0.0.1/x"])
def test_fetch_rejects_non_public_targets(url):
    """Scheme and private-address rejections fail fast with no retries."""
    with pytest.raises(ValueError):
        acquisition.fetch_public(url)
