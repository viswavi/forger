"""Exercise direct generation through real LiteLLM localhost HTTP requests."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pyforger.acquire.model import Model


@pytest.fixture
def endpoint():
    """Serve deterministic OpenAI protocol responses without mocking clients."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            if self.path == "/v1/responses":
                prompt = body["input"][-1]["content"]
                response = {
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 1,
                    "model": "gpt-4o",
                    "status": "completed",
                    "output": [
                        {
                            "id": "rs_test",
                            "type": "reasoning",
                            "summary": [{"type": "summary_text", "text": "thinking"}],
                        },
                        {
                            "id": "msg_test",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": prompt,
                                    "annotations": [],
                                },
                                {"type": "output_text", "text": "!", "annotations": []},
                            ],
                        },
                    ],
                    "usage": {
                        "input_tokens": 7,
                        "output_tokens": 5,
                        "total_tokens": 12,
                        "output_tokens_details": {"reasoning_tokens": 3},
                    },
                }
            else:
                response = {
                    "id": "chatcmpl_test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "chat"},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 5,
                        "total_tokens": 12,
                    },
                }
            if self.path == "/v1/responses" and prompt == "empty":
                response["output"] = []
                response.pop("usage")
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("progress", [False, True])
def test_responses_direct_generation(endpoint, batch, progress):
    """Single and concurrent requests map settings and normalize metadata."""
    url, requests = endpoint
    model = Model(
        "openai/o3",
        api_key="local-test",
        api_base=url,
        temperature=1.0,
        api_mode="responses",
        max_tokens=100,
        reasoning_effort="low",
        extra_body={"custom": True},
        capability_overrides={"vision": False},
        timeout=5,
        num_retries=0,
        show_progress=progress,
    )
    prompts = [
        "first",
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "second"},
        ],
    ]
    results = model.generate_batch(prompts) if batch else [model.generate("first")]
    assert [list(result) for result in results] == (
        [["first!"], ["second!"]] if batch else [["first!"]]
    )
    for result in results:
        assert result.reasoning == ["thinking"]
        assert result.usage.prompt_tokens == 7
        assert result.usage.completion_tokens == 5
        assert result.usage.reasoning_tokens == 3
        assert result.raw.output
        assert result.model_id == "openai/o3"
    assert model.history == results
    assert model.total_usage.prompt_tokens == 7 * len(results)
    for path, body in requests:
        assert path == "/v1/responses"
        assert body["max_output_tokens"] == 100
        assert body["reasoning"] == {"effort": "low"}
        assert body["temperature"] == 1.0
        assert body["custom"] is True
        assert (
            not {
                "api_mode",
                "capability_overrides",
                "n",
                "stop",
                "max_tokens",
                "reasoning_effort",
            }
            & body.keys()
        )


@pytest.mark.parametrize("mode", ["auto", "chat"])
@pytest.mark.parametrize("batch", [False, True])
def test_chat_does_not_leak_internal_settings(endpoint, mode, batch):
    """Chat keeps its endpoint and does not forward Forger-only settings."""
    url, requests = endpoint
    model = Model(
        "openai/gpt-4o",
        api_key="local-test",
        api_base=url,
        api_mode=mode,
        capability_overrides={"vision": False},
        timeout=5,
        num_retries=0,
    )
    results = model.generate_batch(["a", "b"]) if batch else [model.generate("a")]
    assert all(result == ["chat"] for result in results)
    for path, body in requests:
        assert path == "/v1/chat/completions"
        assert not {"api_mode", "capability_overrides"} & body.keys()


@pytest.mark.parametrize("kwargs", [{"n": 2}, {"stop": ["end"]}])
def test_responses_rejects_unsupported_parameters(endpoint, kwargs):
    """Unsupported sampling controls fail before making an HTTP request."""
    url, requests = endpoint
    model = Model(
        "openai/gpt-4o", api_key="local-test", api_base=url, api_mode="responses"
    )
    with pytest.raises(ValueError, match="Responses API"):
        model.generate("a", **kwargs)
    assert requests == []


def test_responses_missing_optional_metadata(endpoint):
    """Empty output and absent usage retain the GenerateResult contract."""
    url, requests = endpoint
    model = Model(
        "openai/gpt-4o",
        api_key="local-test",
        api_base=url,
        api_mode="responses",
        num_retries=0,
        timeout=5,
    )
    result = model.generate("empty")
    assert result == [""]
    assert result.reasoning == [None]
    assert result.usage.prompt_tokens == 0
    assert result.usage.completion_tokens == 0
    assert result.usage.reasoning_tokens == 0
    assert len(requests) == 1


@pytest.mark.parametrize("mode", ["chat", "responses"])
def test_real_native_cost_and_cache_metadata(endpoint, mode):
    base, _ = endpoint
    model = Model("openai/gpt-4o", api_key="test", api_base=base, api_mode=mode)
    result = model.generate("hello")
    assert result.usage.prompt_tokens == 7
    assert result.usage.cost_usd > 0
    assert model.total_cost == result.usage.cost_usd
    model.reset_history()
    assert model.total_cost == 0
    assert model.total_usage.cost_usd == 0


@pytest.mark.parametrize(
    "details,legacy,expected",
    [
        ({"cache_creation_tokens": 4}, 9, 4),
        ({"cache_creation_tokens": 0}, 9, 0),
        ({}, 9, 9),
    ],
)
def test_native_cache_write_canonical_details_and_legacy_fallback(
    details, legacy, expected
):
    """Real LiteLLM usage retains canonical cache writes, including explicit zero."""
    from litellm import ModelResponse
    from litellm.types.utils import Usage as LiteLLMUsage
    from pyforger.acquire.model import _extract_usage

    native_usage = LiteLLMUsage(
        prompt_tokens=10, completion_tokens=5, prompt_tokens_details=details
    )
    # LiteLLM's constructor overwrites canonical details if both are supplied.
    # Assign the legacy field afterward to exercise canonical precedence.
    response = ModelResponse(usage=native_usage)
    response.usage.cache_creation_input_tokens = legacy
    if "cache_creation_tokens" in details:
        response.usage.prompt_tokens_details.cache_creation_tokens = details[
            "cache_creation_tokens"
        ]
    usage = _extract_usage(response)
    assert usage.cache_write_tokens == expected
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
