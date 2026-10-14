"""Real subprocess, SDK, and optional Docker runtime tests (no mocked execution)."""

import json
import os
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from pyforger.acquire import Model
from pyforger.agent import CONDENSER_KEEP_FIRST, CONDENSER_MAX_TOKENS
from pyforger.agent import AgentExecutionError as RuntimeFailure
from pyforger.agent import redact as _redact
from pyforger.agent import run_agent


def _sdk_run(prompt, model, base_url, events, iterations, **options):
    return run_agent(
        prompt,
        Model(
            model,
            api_key=os.environ.get("LLM_API_KEY"),
            api_base=base_url,
            temperature=None,
            reasoning_effort=None,
            disable_prompt_cache_key=True,
            capability_overrides={
                "supports_vision": True,
                "supports_prompt_cache_key": False,
            },
        ),
        events,
        iterations,
        **options,
    )


@contextmanager
def completion_server(message, host="127.0.0.1"):
    """Protocol fixture replaces only the external paid API; the SDK runs normally."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        """Verify Handler."""

        def do_POST(self):
            """Verify do POST."""
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
            """Verify log message."""
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


class RuntimeTests(unittest.TestCase):
    """Exercise real SDK conversations against a local protocol fixture."""

    def test_real_sdk_no_tools_and_full_events(self):
        """Verify real sdk no tools and full events."""
        with tempfile.TemporaryDirectory() as folder:
            events = Path(folder) / "events.jsonl"
            with completion_server({"role": "assistant", "content": "complete"}) as (
                url,
                requests,
            ):
                self.assertEqual(
                    _sdk_run("hello", "openai/test", url, events, 2), "complete"
                )
            self.assertTrue(requests)
            self.assertFalse(requests[0].get("tools"))
            self.assertNotIn("prompt_cache_key", requests[0])
            self.assertEqual(requests[0]["reasoning_effort"], "high")
            payloads = [json.loads(line) for line in events.read_text().splitlines()]
            self.assertTrue(
                any(
                    p.get("kind") == "MessageEvent" and p.get("source") == "agent"
                    for p in payloads
                )
            )
            self.assertNotIn("test-not-a-real-secret", events.read_text())

    def test_sdk_result_validator_requests_one_correction(self):
        """Verify sdk result validator requests one correction."""
        messages = [
            {"role": "assistant", "content": "invalid"},
            {"role": "assistant", "content": '{"status":"valid"}'},
        ]

        def validate(text):
            """Verify validate."""
            if text != '{"status":"valid"}':
                raise RuntimeFailure("wrong schema")

        with tempfile.TemporaryDirectory() as folder:
            events = Path(folder) / "events.jsonl"
            with completion_server(messages) as (url, requests):
                result = _sdk_run(
                    "Return JSON",
                    "openai/test",
                    url,
                    events,
                    2,
                    result_validator=validate,
                )
            self.assertEqual(result, '{"status":"valid"}')
            self.assertEqual(len(requests), 2)
            self.assertIn("strict result parser", str(requests[1]))
            self.assertIn("invalid", events.read_text())
            self.assertIn('{\\"status\\":\\"valid\\"}', events.read_text())

    def test_condenser_keeps_task_prompt(self):
        # Event 0 is the system prompt and event 1 is the task prompt that
        # carries the review contract; condensing must never drop the latter, or
        # the reviewer loses the criterion names and output schema.
        """Verify condenser keeps task prompt."""
        self.assertGreaterEqual(CONDENSER_KEEP_FIRST, 2)

    def test_condenser_window_holds_review_working_set(self):
        # Too small a window evicts evidence the reviewer just read, so it
        # re-reads it until the iteration budget is exhausted.
        """Verify condenser window holds review working set."""
        self.assertGreaterEqual(CONDENSER_MAX_TOKENS, 24000)

    def test_secret_redaction(self):
        """Verify secret redaction."""
        name = "PAPER2RL_TEST_SECRET"
        previous = os.environ.get(name)
        try:
            os.environ[name] = "sensitive-test-value"
            self.assertEqual(_redact("a sensitive-test-value b"), "a [REDACTED] b")
        finally:
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous

    def test_sdk_api_failure_real_connection(self):
        """Verify sdk api failure real connection."""
        previous = os.environ.get("LLM_API_KEY")
        try:
            os.environ["LLM_API_KEY"] = "test-not-a-real-secret"
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "events.jsonl"
                with self.assertRaises(Exception) as caught:
                    _sdk_run(
                        "Say hello", "openai/test", "http://127.0.0.1:1/v1", path, 1
                    )
                self.assertNotIsInstance(caught.exception, TypeError)
                self.assertTrue(path.exists())
                self.assertNotIn("test-not-a-real-secret", path.read_text())
        finally:
            if previous is None:
                os.environ.pop("LLM_API_KEY", None)
            else:
                os.environ["LLM_API_KEY"] = previous


if __name__ == "__main__":
    unittest.main()


def test_correction_is_validated_again(tmp_path):
    """Verify correction is validated again."""
    with completion_server(
        [
            {"role": "assistant", "content": "invalid"},
            {"role": "assistant", "content": "still invalid"},
        ]
    ) as (url, requests):

        def validate(text):
            """Verify validate."""
            raise ValueError("invalid result")

        with pytest.raises(ValueError):
            _sdk_run(
                "Return valid JSON",
                "openai/test",
                url,
                tmp_path / "events.jsonl",
                2,
                result_validator=validate,
            )
        assert len(requests) == 2


def test_agent_rejects_conflicting_capabilities_before_calls(tmp_path):
    """Verify agent rejects conflicting capabilities before calls."""
    with pytest.raises(ValueError, match="Choose"):
        run_agent(
            "prompt",
            None,
            tmp_path / "events.jsonl",
            1,
            evidence={"input": "data"},
            tools=[object()],
        )
    with pytest.raises(ValueError, match="attempts"):
        run_agent("prompt", None, tmp_path / "events.jsonl", 1, attempts=0)
