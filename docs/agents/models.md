# Models, prompts, and credentials

## The `Model` class

```python
import os
from pyforger.acquire import Model

model = Model(
    os.environ.get("MODEL_ID"),
    api_key=os.environ.get("LITELLM_API_KEY"),
    temperature=1.0,
    max_tokens=None,
    num_retries=2,
)
```

For gateway models (ids starting with `litellm_proxy/`), `api_key` defaults to `LITELLM_API_KEY` and `api_base` to
`LITELLM_BASE_URL`. The default temperature is 1.0, which reasoning models such as DeepSeek-V4 need; at temperature
0 they can loop until the request times out.

Use `Model(..., reasoning_effort="low")` (or `"none"`, `"high"`) to set reasoning effort. `Model` sends it in the
form each model reads: DeepSeek-V4 as `reasoning={"effort": "low"}` (the gateway ignores `reasoning_effort` for it),
GLM-5.3 and other models as `reasoning_effort`. It overrides any reasoning setting passed through `**kwargs` or
`extra_body`. Without it, DeepSeek-V4-Flash on the gateway does not think at all.
`dynamic_step` forwards the reasoning effort and `extra_body` to OpenHands.

## Prompt types

Two type aliases are exported from `pyforger.acquire.model`:

```python
Message = dict[str, str]                 # {"role": "...", "content": "..."}
Prompt  = Union[str, list[Message]]      # string or full chat message list
```

A bare string is wrapped as a single user message. Pass a `list[Message]` for a system prompt or multi-turn structure.

For operations that prompt per row (e.g. `generate_from_context`), the template type is:

```python
PromptTemplate = Union[str, Callable[[dict], Prompt]]
```

A string template is filled with `str.format_map(item)` (use `{field_name}` placeholders). A callable receives the row
dict and must return a `Prompt`.
