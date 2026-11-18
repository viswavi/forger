"""Direct LiteLLM helpers hold each call's worst case until it reports back."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coding"))

from rl import token_budget  # noqa: E402

DEEPSEEK = "litellm_proxy/azure_ai/DeepSeek-V4.1-Flash"


class _Timeout(Exception):
    status_code = 408


class HoldTests(unittest.TestCase):
    """`completion` holds, then records or charges, each call."""

    def setUp(self):
        """Point the generation ledger at a temporary file."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        env = {
            "FORGER_TOKEN_USAGE_FILE": str(self.dir / "usage.json"),
            "FORGER_TOKEN_UNIT_LIMIT": "1000000",
            "LITELLM_API_KEY": "test",
            "LITELLM_BASE_URL": "http://gateway",
        }
        environ = patch.dict(os.environ, env)
        environ.start()
        self.addCleanup(environ.stop)

    def ledger(self):
        """The ledger's current state."""
        return token_budget.TokenUnitBudget(self.dir / "usage.json", 1000000).snapshot()

    def call(self, **kwargs):
        """One small generation call through the helper."""
        return token_budget.completion(
            model=DEEPSEEK,
            messages=[{"role": "user", "content": "abcd"}],
            max_tokens=100,
            **kwargs,
        )

    def test_timeout_is_charged_at_its_hold(self):
        """A timed-out call may still be billed, so it costs 2 + 100 x 4 units."""
        with patch("litellm.completion", side_effect=_Timeout("timed out")):
            with self.assertRaises(_Timeout):
                self.call()
        state = self.ledger()
        self.assertEqual((state["token_units"], state["unreturned_calls"]), (402, 1))
        self.assertEqual(state["held_token_units"], 0)

    def test_response_replaces_the_hold(self):
        """A response is charged its actual usage, not its hold."""
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20)
        result = SimpleNamespace(usage=usage, _hidden_params={"response_cost": 0.0})
        with patch("litellm.completion", return_value=result):
            self.call()
        state = self.ledger()
        self.assertEqual((state["token_units"], state["held_token_units"]), (90, 0))

    def test_no_max_tokens_is_sent_uncapped_and_held_at_65536(self):
        """With no max_tokens the call is uncapped, as in Forger; it holds 65,536."""
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20)
        result = SimpleNamespace(usage=usage, _hidden_params={"response_cost": 0.0})
        seen = {}

        def reply(**kwargs):
            seen["max_tokens"] = kwargs.get("max_tokens")
            seen["held"] = self.ledger()["held_token_units"]
            return result

        with patch("litellm.completion", side_effect=reply):
            token_budget.completion(
                model=DEEPSEEK, messages=[{"role": "user", "content": "abcd"}]
            )
        self.assertEqual(seen, {"max_tokens": None, "held": 2 + 65_536 * 4})

    def test_sdk_retries_are_refused(self):
        """A retried attempt would be billed but never recorded."""
        with self.assertRaisesRegex(ValueError, "retry yourself"):
            self.call(num_retries=2)


if __name__ == "__main__":
    unittest.main()
