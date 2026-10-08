"""Local inference server readiness and exclusive-port preflight."""
import json
import os
import socket
import sys
from urllib.request import Request, urlopen


def main():
    """Run the command-line utility described in the module documentation."""
    action, port = sys.argv[1], int(sys.argv[2])
    if action == "free":
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
        return
    model = sys.argv[3]
    request = Request(
        f"http://127.0.0.1:{port}/v1/models",
        headers={"Authorization": "Bearer " + os.environ.get("VLLM_API_KEY", "tok")},
    )
    with urlopen(request, timeout=2) as response:
        models = json.load(response)["data"]
    if model not in {row["id"] for row in models}:
        raise ValueError("Inference server is serving a different model")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)
