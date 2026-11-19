"""Call Claude through Forger's Model class, without any pipeline.

Every Forger op (sample_from_LLM, generate_from_context, the LLM filters) takes a
Model. This script shows what a Model does on its own: one prompt, several samples,
a batch of prompts, a system prompt, and the cost of it all.

    python claude_model.py      # well under a cent

The key comes from ANTHROPIC_API_KEY, here read from a .env file.
"""
from dotenv import load_dotenv

from pyforger.acquire import Model

load_dotenv()

# A LiteLLM model id: "anthropic/" plus the Claude model name. No base URL needed.
# max_concurrency caps the requests in flight; reasoning_effort sets how long it thinks.
haiku = Model("anthropic/claude-haiku-5-5", max_concurrency=8, reasoning_effort="low")

# 1. One prompt, one answer. The result is a list of completions.
result = haiku.generate("In one sentence, why is the sky blue?")
print("1.", result[0])
print(
    f"   cost ${result.usage['cost_usd']:.5f}, finish reason {result.finish_reasons[0]}"
)

# 2. Several samples of the same prompt (sent as concurrent requests).
names = haiku.generate("Name one moon of Jupiter. Reply with the name only.", n=4)
print("2.", names.texts)

# 3. A batch of prompts, sent at once: one result per prompt, in order.
planets = ["Mercury", "Venus", "Mars"]
results = haiku.generate_batch(
    [f"How many moons does {p} have? Reply with a number only." for p in planets]
)
for planet, r in zip(planets, results):
    print(f"3. {planet}: {r[0]}")

# 4. A system prompt: pass chat messages instead of a string.
pirate = haiku.generate(
    [
        {"role": "system", "content": "You answer like a pirate, in one sentence."},
        {"role": "user", "content": "What is a light-year?"},
    ],
    max_tokens=2000,
)
print("4.", pirate[0])

# 5. The model keeps a running total of everything it spent.
print(f"5. total ${haiku.total_cost:.5f} for {haiku.total_usage}")
