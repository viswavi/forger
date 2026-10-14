"""Local scripted completion API for real SDK integration tests."""

import json
import os
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@contextmanager
def completion_server(message, host="127.0.0.1"):
    """Protocol fixture replaces only the external paid API; the SDK runs normally."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        """Serve scripted OpenAI-compatible completion messages."""

        def do_POST(self):
            """Record the request and return the next scripted message."""
            requests.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            response_message = (
                message[len(requests) - 1] if isinstance(message, list) else message
            )
            payload = json.dumps(
                {
                    "id": "chatcmpl-runtime-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test",
                    "choices": [
                        {
                            "index": 0,
                            "message": response_message,
                            "finish_reason": "tool_calls"
                            if response_message.get("tool_calls")
                            else "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            """Keep test output free of HTTP access logs."""
            pass

    server = ThreadingHTTPServer((host, 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    previous = os.environ.get("LLM_API_KEY")
    os.environ["LLM_API_KEY"] = "test-not-a-real-secret"
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        if previous is None:
            os.environ.pop("LLM_API_KEY", None)
        else:
            os.environ["LLM_API_KEY"] = previous
