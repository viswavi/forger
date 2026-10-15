"""In-flight reservations keep concurrent Tinker requests under the dollar limit.

Fake samplers only: no Tinker or network calls.
"""

import asyncio
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from coding.rl import tinker_budget
from coding.rl.tinker_budget import (
    TinkerBudget,
    TinkerBudgetExceeded,
    cost_usd,
    sample_tokens,
)

ROOT = Path(__file__).resolve().parents[1]
QWEN = "Qwen/Qwen3-8B"
PROMPT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
MAX_TOKENS = 1000
# What one request may cost: its prompt plus MAX_TOKENS sampled tokens.
WORST = cost_usd(QWEN, sample_tokens(len(PROMPT), 1, 0, [MAX_TOKENS]))


class FakeClient:
    """A Tinker sampling client whose requests finish when the test says so.

    Each request checks that recorded spend plus every in-flight worst case
    stays within the ledger's limit at the moment it starts.
    """

    def __init__(self, budget, sampled=MAX_TOKENS):
        """Charge `budget`; every completion has `sampled` tokens."""
        self.budget, self.sampled = budget, sampled
        self.active = self.peak = self.started = 0
        self.release = asyncio.Event()  # cleared: requests wait for the test
        self.failures = 0  # how many of the next requests to finish fail

    async def sample_async(self, prompt, num_samples, sampling_params):
        """Wait for the test to release requests, then return one completion."""
        self.active += 1
        self.started += 1
        self.peak = max(self.peak, self.active)
        spent = self.budget.snapshot()["cost_usd"]
        assert spent + self.active * WORST <= self.budget.limit + 1e-12
        try:
            await self.release.wait()
            if self.failures:
                self.failures -= 1
                raise RuntimeError("sampling failed")
        finally:
            self.active -= 1
        sequence = SimpleNamespace(tokens=[0] * self.sampled, stop_reason="stop")
        return SimpleNamespace(sequences=[sequence], prompt_cache_hit_tokens=0)


class TinkerChatReservationTests(unittest.TestCase):
    """TinkerChat holds each request's worst case from check until record."""

    def setUp(self):
        """Use a scratch directory for ledgers."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)
        # tinker_sampling imports rl.tinker_budget from coding/, as the
        # evaluator does; drop those modules again afterwards.
        modules = patch.dict(sys.modules)
        modules.start()
        self.addCleanup(modules.stop)
        path = patch.object(sys, "path", [str(ROOT / "coding"), *sys.path])
        path.start()
        self.addCleanup(path.stop)
        self.sampling = importlib.import_module("tinker_sampling")
        self.budget_module = importlib.import_module("rl.tinker_budget")

    def chat(self, limit_requests, sampled=MAX_TOKENS):
        """A TinkerChat (no SDK) whose ledger fits `limit_requests` worst cases."""
        budget = self.budget_module.TinkerBudget(
            self.tmp / "sft.json", WORST * limit_requests, name="sft"
        )
        chat = object.__new__(self.sampling.TinkerChat)
        chat.budget, chat.session_id, chat.base_model = budget, "s1", QWEN
        chat.types = SimpleNamespace(
            SamplingParams=lambda **kwargs: kwargs,
            ModelInput=SimpleNamespace(from_ints=list),
        )
        chat.tokenizer = SimpleNamespace(
            apply_chat_template=lambda messages, **kwargs: "prompt",
            encode=lambda text, add_special_tokens: PROMPT,
            decode=lambda tokens, skip_special_tokens: "x" * len(tokens),
        )
        chat.args = SimpleNamespace(
            enable_thinking=None,
            max_tokens=MAX_TOKENS,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        )
        chat.stop, chat.max_context = [0], 32768
        chat.client = FakeClient(budget, sampled)
        return chat

    async def start(self, chat, count):
        """Start `count` requests and let each one either sample or wait."""
        tasks = [
            asyncio.create_task(chat([{"role": "user", "content": str(i)}]))
            for i in range(count)
        ]
        for _ in range(20):
            await asyncio.sleep(0)
        return tasks

    def test_concurrent_requests_stay_under_the_limit(self):
        """Ten concurrent worst-case requests with room for 3.5: three run."""
        chat = self.chat(3.5)

        async def run():
            tasks = await self.start(chat, 10)
            self.assertEqual(chat.client.started, 3)
            chat.client.release.set()
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = asyncio.run(run())
        refused = [r for r in results if isinstance(r, BaseException)]
        self.assertEqual(len(refused), 7)
        self.assertTrue(
            all(isinstance(r, self.budget_module.TinkerBudgetExceeded) for r in refused)
        )
        state = chat.budget.snapshot()
        self.assertEqual(state["calls"], 3)
        self.assertLessEqual(state["cost_usd"], chat.budget.limit)
        self.assertEqual(chat.client.peak, 3)
        self.assertEqual(self.budget_module._RESERVED, {})

    def test_short_completions_let_waiting_requests_run(self):
        """A request that does not fit waits for others to record, then runs."""
        # Each request samples a tenth of its worst case, so after three are
        # recorded the waiting ones fit again.
        chat = self.chat(3.5, sampled=MAX_TOKENS // 10)

        async def run():
            tasks = await self.start(chat, 10)
            # Three are sampling; the rest wait instead of failing.
            self.assertEqual(chat.client.started, 3)
            self.assertFalse(any(task.done() for task in tasks))
            chat.client.release.set()
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = asyncio.run(run())
        self.assertFalse([r for r in results if isinstance(r, BaseException)])
        state = chat.budget.snapshot()
        self.assertEqual(state["calls"], 10)
        self.assertLessEqual(state["cost_usd"], chat.budget.limit)
        self.assertEqual(self.budget_module._RESERVED, {})

    def test_failed_request_releases_its_reservation(self):
        """A failed request frees its worst case for the one waiting on it."""
        chat = self.chat(1.5)
        chat.client.failures = 1

        async def run():
            first, second = await self.start(chat, 2)
            self.assertEqual(chat.client.started, 1)  # the second one waits
            chat.client.release.set()
            with self.assertRaisesRegex(RuntimeError, "sampling failed"):
                await first
            return await second

        completion, _, usage = asyncio.run(run())
        self.assertEqual(len(completion), MAX_TOKENS)
        self.assertEqual(usage["sampled_tokens"], MAX_TOKENS)
        self.assertEqual(chat.budget.snapshot()["calls"], 1)
        self.assertEqual(self.budget_module._RESERVED, {})

    def test_sequential_requests_are_checked_against_recorded_spend(self):
        """One request at a time behaves as before: refused once spend is too high."""
        chat = self.chat(2.5)
        chat.client.release.set()

        async def run():
            for _ in range(2):
                await chat([{"role": "user", "content": "q"}])
            with self.assertRaisesRegex(
                self.budget_module.TinkerBudgetExceeded, "sft budget"
            ):
                await chat([{"role": "user", "content": "q"}])

        asyncio.run(run())
        self.assertEqual(chat.budget.snapshot()["calls"], 2)
        self.assertEqual(chat.client.started, 2)


class ReserveTests(unittest.TestCase):
    """TinkerBudget.reserve on its own."""

    def setUp(self):
        """Use a scratch directory for ledgers."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)

    def test_unlimited_ledger_never_waits(self):
        """A record-only ledger reserves without limiting."""
        budget = TinkerBudget(self.tmp / "eval.json", None, "eval")

        async def run():
            async with budget.reserve(10**6), budget.reserve(10**6):
                self.assertEqual(tinker_budget._RESERVED[budget.path], (2, 2e6))

        asyncio.run(run())
        self.assertEqual(tinker_budget._RESERVED, {})

    def test_cancelled_waiter_leaves_nothing_reserved(self):
        """Cancelling a waiting request neither reserves nor blocks later ones."""
        budget = TinkerBudget(self.tmp / "sft.json", 1.0, name="sft")

        async def run():
            entered = asyncio.Event()
            finish = asyncio.Event()

            async def holder():
                async with budget.reserve(0.8):
                    entered.set()
                    await finish.wait()
                    budget.record({"sample": 1_000_000})  # $0.60

            async def waiter():
                async with budget.reserve(0.5):
                    pass

            held = asyncio.create_task(holder())
            await entered.wait()
            waiting = asyncio.create_task(waiter())
            await asyncio.sleep(0)
            waiting.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiting
            finish.set()
            await held
            # $0.60 recorded, nothing in flight: $0.40 fits, $0.50 is refused.
            async with budget.reserve(0.4):
                pass
            with self.assertRaises(TinkerBudgetExceeded):
                async with budget.reserve(0.5):
                    pass

        asyncio.run(run())
        self.assertEqual(tinker_budget._RESERVED, {})
        self.assertEqual(budget.snapshot()["calls"], 1)

    def test_rl_sample_groups_stay_under_the_limit(self):
        """RL rollouts reserve each group's worst case too."""
        from coding.rl import tinker_training

        tokenizer = SimpleNamespace(
            apply_chat_template=lambda *args, **kwargs: PROMPT,
            convert_tokens_to_ids=lambda token: 0,
        )
        worst = cost_usd(QWEN, sample_tokens(len(PROMPT), 4, 0, [MAX_TOKENS] * 4))
        budget = TinkerBudget(self.tmp / "rl.json", worst * 2.5, name="rl")
        state = {"active": 0, "peak": 0}

        class Sampler:
            async def sample_async(self, **kwargs):
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
                await asyncio.sleep(0)
                state["active"] -= 1
                sequence = SimpleNamespace(
                    tokens=[0] * MAX_TOKENS, stop_reason="length"
                )
                return SimpleNamespace(
                    sequences=[sequence] * 4, prompt_cache_hit_tokens=0
                )

        async def run():
            return await asyncio.gather(
                *[
                    tinker_training.sample_group(
                        Sampler(),
                        tokenizer,
                        [],
                        budget,
                        num_samples=4,
                        max_tokens=MAX_TOKENS,
                        temperature=1.0,
                    )
                    for _ in range(6)
                ],
                return_exceptions=True,
            )

        results = asyncio.run(run())
        self.assertEqual(sum(isinstance(r, TinkerBudgetExceeded) for r in results), 4)
        self.assertEqual(state["peak"], 2)
        self.assertEqual(budget.snapshot()["calls"], 2)
        self.assertLessEqual(budget.snapshot()["cost_usd"], budget.limit)
        self.assertEqual(tinker_budget._RESERVED, {})


if __name__ == "__main__":
    unittest.main()
