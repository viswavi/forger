"""Measure prompt-brittleness vs underlying ability for ONE checkpoint:
  eval = real inspect GSM8K 10-shot prompt (ends 'Reasoning:')   -> brittleness
  bare = single 0-shot question (the trained format)             -> raw ability
Prints one ROW line. Usage: python trajectory.py <model_path> <label>"""
import os, re, sys, json, hashlib
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
from datasets import load_dataset
import random

MPATH, LABEL = sys.argv[1], sys.argv[2]
N=-1
if N > 0:
    test = load_dataset("openai/gsm8k", "main", split=f"test[:{N}]")
else:
    test = load_dataset("openai/gsm8k", "main", split=f"test")
    N = "full"

QS = [(r["question"], r["answer"].split("####")[-1].strip().replace(",", "")) for r in test]
train = load_dataset("openai/gsm8k", "main", split="train")
def eval_user(q):
    return ('Solve the following math problem step by step. The last line of your response should be of the form '
            '"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\n'
            f'{q}\n\nRemember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" '
            '(without quotes) where $ANSWER is the answer to the problem, and you do not need to use a '
            '\\boxed command.\n\nReasoning:')

TEMPLATE = open("/home/vijayv/PostTrainBench/src/eval/templates/qwen3.jinja").read()


def last_number(t):
    first_number = None
    if "ANSWER:" in t:
        # find segments that follow "ANSWER:"
        answer_candidates = re.findall(r"ANSWER:\s*(.*)", t)
        # find first number in those segments
        for candidate in answer_candidates:
            n = re.findall(r"-?\d[\d,]*\.?\d*", candidate.replace(",", ""))
            if n:
                first_number = n[0].rstrip(".")
                break
    if first_number is not None:
        return first_number.replace(",", "").strip()
    else:
        n = re.findall(r"-?\d[\d,]*\.?\d*", t.replace(",", ""))
        return n[0].rstrip(".") if n else None


def degenerate(t):
    if not t.strip():
        return True
    na = sum(1 for c in t if ord(c) > 127) / max(len(t), 1)
    tk = t.split()
    return na > 0.08 or (len(tk) > 25 and len(set(tk)) < 0.25 * len(tk))

# Build the prompts/conversations (cheap, no model needed)
convos, origin = [], []
for q, gold in QS:

    sample = train.select(random.Random(42).sample(range(len(train)), 10))
    in_context_examlpes = "\n\n".join(
    f'{r["question"]}\n\nReasoning:\n{r["answer"].split("####")[0].strip()}\n\nANSWER: {r["answer"].split("####")[-1].strip()}'
    for r in sample)
    convos.append([{"role": "system", "content": in_context_examlpes}, {"role": "user", "content": eval_user(q)}]); origin.append(("eval", gold))

for q, gold in QS:                                   # bare condition
    convos.append([{"role": "user", "content": f'Solve the following math problem. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\n{q}'}]); origin.append(("bare", gold))

# ---- Prediction cache ----
PRED_DIR = f"predictions_cache_gsm8k_N_{N}"
os.makedirs(PRED_DIR, exist_ok=True)
MODEL_STR_ESCAPED = re.sub(r"[^A-Za-z0-9._-]", "_", MPATH)
CACHE_PATH = os.path.join(PRED_DIR, MODEL_STR_ESCAPED)

# A stable key per prompt so we can verify the cache covers the whole test set.
def prompt_key(cond, convo):
    prompt = convo[-1]["content"]
    return prompt

keys = [prompt_key(cond, convo) for (cond, _gold), convo in zip(origin, convos)]

texts = None
if os.path.exists(CACHE_PATH):
    try:
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        cached = cache.get("predictions", {})
        if all(k in cached for k in keys):
            texts = [cached[k] for k in keys]
    except (json.JSONDecodeError, OSError):
        texts = None

if texts is None:
    from vllm import LLM, SamplingParams
    llm = LLM(model=MPATH, dtype="bfloat16", gpu_memory_utilization=0.6, max_model_len=4096)
    outs = llm.chat(convos, SamplingParams(temperature=0.0, max_tokens=512), chat_template=TEMPLATE, use_tqdm=False)
    texts = [o.outputs[0].text for o in outs]
    cache = {"model": MPATH, "predictions": {k: t for k, t in zip(keys, texts)}}
    tmp_path = CACHE_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(cache, f, indent=4, ensure_ascii=False)
    os.replace(tmp_path, CACHE_PATH)

agg = {"eval": [0, 0, 0], "bare": [0, 0, 0]}  # correct, deg, n
for (cond, gold), t in zip(origin, texts):
    a = agg[cond]; a[2] += 1
    try:
        if degenerate(t):
            a[1] += 1
        elif last_number(t) and gold and abs(float(last_number(t)) - float(gold)) < 1e-6:
            a[0] += 1
    except:
        breakpoint()


e, b = agg["eval"], agg["bare"]
print(f"ROW {LABEL:12} eval_acc={e[0]/e[2]:.2f} eval_degen={e[1]/e[2]:.2f}  bare_acc={b[0]/b[2]:.2f} bare_degen={b[1]/b[2]:.2f}")
