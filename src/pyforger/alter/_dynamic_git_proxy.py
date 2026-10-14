"""A public-address-only, transfer-limited CONNECT proxy for Git clones."""

from __future__ import annotations

import selectors
import socket
import socketserver
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

_MAX_TRANSFER = 32 * 1024 * 1024
_ALLOWED_HOSTS = {"github.com", "gitlab.com"}


@contextmanager
def git_proxy() -> Iterator[str]:
    """Yield a local proxy URL that pins validated Git hosts to public IPs."""
    # Import lazily to avoid a module cycle with resource preparation.
    from pyforger.alter._dynamic_resources import _public_https_address

    total = 0
    lock = threading.Lock()

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            """Accept CONNECT for the two supported hosts and relay bounded bytes."""
            nonlocal total
            try:
                line = self.rfile.readline(4096).decode("ascii")
                method, authority, _ = line.split(" ", 2)
                for _ in range(64):
                    if self.rfile.readline(4096) in (b"\r\n", b"\n", b""):
                        break
                host, separator, port = authority.rpartition(":")
                if (
                    method != "CONNECT"
                    or not separator
                    or host.lower() not in _ALLOWED_HOSTS
                    or port != "443"
                ):
                    self.connection.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
                    return
                _, address = _public_https_address(f"https://{host.lower()}")
                with socket.create_connection((address, 443), timeout=5) as remote:
                    self.connection.sendall(
                        b"HTTP/1.1 200 Connection Established\r\n\r\n"
                    )
                    with selectors.DefaultSelector() as selector:
                        selector.register(self.connection, selectors.EVENT_READ, remote)
                        selector.register(remote, selectors.EVENT_READ, self.connection)
                        deadline = time.monotonic() + 60
                        while time.monotonic() < deadline:
                            ready = selector.select(timeout=1)
                            for key, _ in ready:
                                source = key.fileobj
                                if not isinstance(source, socket.socket):
                                    raise TypeError("Proxy selector requires sockets")
                                destination = (
                                    remote
                                    if source is self.connection
                                    else self.connection
                                )
                                data = source.recv(64 * 1024)
                                if not data:
                                    return
                                with lock:
                                    total += len(data)
                                    if total > _MAX_TRANSFER:
                                        return
                                destination.sendall(data)
            except (OSError, UnicodeError, ValueError):
                return

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with Server(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            thread.join(timeout=2)
