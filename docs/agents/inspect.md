# `pyforger.inspect`

Inspect ops return a *report* about rows and never change them. Signatures are in [OPS.md](../OPS.md#inspect).

| Export | Cached | Use it to |
|---|---|---|
| `topic_based_clustering` | yes | Label each row with one of `num_clusters` LLM-named topics for one field. |
| `eval_tree` | yes | Build a capability tree and weakness profile from scored model outputs. |
| `inspect_step` | no | See more rows of a step: reprints its latest full report (10 rows); `rows=True` returns all. |
| `spend` | no | Print and return the spend ledger's totals by run, pipeline, and step. |
| `runs` | no | Print and return the saved `@pipeline` runs with step yields, e.g. `runs("main")`. |
| `phrase_frequency` | no | Print, per string field, the phrases and names the most rows share. No LLM. |
| `student_log_likelihood` | yes | How likely the student finds each row's target, in bits, beside its own responses. |
| `rubric_scores` | grades | Print and return a reward's score histogram on responses, samples, and controls. |
| `embedding_dispersion` | yes | Print and return how dispersed a field is in embedding space vs. a reference. No LLM. |

Run `eval_tree` with `reasoning_effort="none"` (`"low"` cost 15 times as much and 10% of annotations ran out of
tokens in reasoning loops); `topic_based_clustering` with `"low"` or `"none"` (higher efforts hit gateway timeouts).

## `phrase_frequency`

Share of rows containing each phrase or name, e.g. `rewrite  phrases: "i need" 78/79 (99%)`. Topic clustering can
look balanced while every row says "I need"; run both.

## `student_log_likelihood`

Scores each row's target (e.g. `"<think>...</think>answer"`) under the student, a `LocalModel`, as SFT trains on it:
bits (`-log2` of each token's probability) per token and in total, reasoning and answer apart. For scale, the
student also answers `own_samples` prompts at temperature 1 and its own responses are scored the same way. A coding
SFT set for Qwen3-4B:

```text
                  rows   median tokens   bits/token   bits/row   total bits
  targets          134             736         3.77      2,641   450.2k
    reasoning      134             600         4.18
    answer         134             126         0.45
  student's own    134           8,064         0.44      4,011   -
  excess over the student's own rate (0.44 bits/token): 397.8k bits in total
```

- *Excess bits*: a target's bits minus the student's own rate times its tokens. About 0 for on-policy text such as
  STaR traces; grows with foreignness and length. The report lists the rows with the most.
- It describes the data; it is not a validated predictor of SFT results. Compare datasets or rows, not one number.
- Sampling the student's own responses is the slow part (134 long responses: 37 minutes on one L40S, 19 on two); a
  `LocalModel` scores and samples on every visible GPU. Cached.

## `rubric_scores`

`rubric_scores(rows, grade, *, n=50, response_fields="response", model=None, k=4, ...)` bins a reward's scores before
RL. `grade(response, row)` returns a score in [0, 1] or None: a code rubric, an LLM judge, or exact match. Columns: each
response field (e.g. the SFT target), `k` samples per prompt from `model` (what RL sees), and three controls no task
should reward: `empty`, `random` (200 random characters), and `other_row` (another row's response). On a rubric that
gave 1.0 to any response without a python code block, the targets looked fine; `empty` did not:

```text
  score          response   sampled ×3   empty   random   other_row
  0                    10           25       0        0          50
  1                    40          125      50       50           0
  mean               0.80         0.83    1.00     1.00        0.00
  sampled: 25/50 prompts gave all samples the same score (0 all 0, 25 all 1): no RL gradient there
  ! empty scores 1.00 on average, at least half of response's 0.80: RL can earn it too
```

- A control earning half the reference column's mean is flagged. Prompts whose samples all score the same give no
  RL gradient.
- `report.model_dump()` is plain JSON: bins, and per column counts, mean, shares at 0 and 1, and per-row scores.
  Grades are cached (`grade` by its `cache_key()`).

## `embedding_dispersion`

`embedding_dispersion(rows, field, *, reference=None, k=5, duplicate_threshold=0.95, ...)` measures how spread out
`rows` are, ideally against a `reference` you want to resemble (e.g. retrieved documents or a seed dataset):

```text
  coverage 0.41  12/29 reference rows have a row within their 5-NN radius (29 random rows of each)
      not covered: "Find the remainder when 8^9 is divided by 7."  nearest 0.06
  off-distribution 0.07  2/29 rows are outside every reference row's 5-NN radius
  near-duplicates 0.17  5/29 rows in 4 pairs with cosine ≥ 0.95 (3 rows repeat an earlier row exactly)
  Vendi 11.4  effective number of distinct rows among 29 (reference 17.5)
```

- *Coverage* ([Naeem et al. 2020](https://arxiv.org/abs/2002.09797)): share of reference rows with a row inside their
  k-NN ball; computed on equal-size samples (two samples of one distribution score about 0.97 at `k=5`).
- *Off-distribution*: share of rows outside every reference row's ball. *Vendi*: effective number of distinct rows.
- *Near-duplicates*: 0.95 counts edited copies of long problems. With Qwen3-Embedding-0.6B, 220-word coding problems
  with whitespace, case, a few numbers, or one sentence changed score 0.95-1.00, and 1 pair among 1,131 distinct
  LiveCodeBench problems reaches 0.95 (5% of rows at 0.9). Use 0.9 for one-line rows: one changed number scores 0.93
  (median), the nearest different problem 0.66.
- Same embedding backends as `topic_based_clustering`. Numbers depend on the embedding: compare datasets under one
  embedding, not one number alone. Without a `reference`, only near-duplicates and Vendi are reported.

## `topic_based_clustering`

Embeddings group rows, an LLM names each group, then assigns each row a topic
([ClusterFusion](https://arxiv.org/abs/2512.04350)). `filter.by_topic_frequency` uses it to downsample topics.

```python
clusters = topic_based_clustering(  # clusters[i] is the topic of rows[i]
    rows, field="problem", model=model, num_clusters=20,
    context="competition math. Topics: subject and technique, NOT the numeric answer.",
)
```

- Pass `context` (what the corpus is, what topics should capture); it improves topic names.
- Default embeddings need `pip install "pyforger[clustering]"`; without it the op falls back to CPU TF-IDF with a
  warning. `embed_fn` plugs in another backend.
- n values embed on `min(n // 500 + 1, visible GPUs)` GPUs (400 on one, 999 on two). Batches hold up to
  `FORGER_EMBED_BATCH_TOKENS` padded tokens (default 16384); a longer text is a batch of its own, never truncated.

## `eval_tree`

[EvalTree](https://arxiv.org/abs/2503.08893) tags each scored row (e.g. validation predictions) with the capability
it needs, groups them into a tree, and gives each node's accuracy with a binomial interval.
`report["weaknesses"]` lists subtrees entirely below `threshold`; with a few hundred rows it is often empty (the op
warns), so read `report["lowest"]` (the five lowest bounds). Only the question is used to infer the capability; pass
`reference_field` to add verified solutions. Rows that get no annotation are listed in `report["unannotated"]`. Run
this with `reasoning_effort="none"` and with a faster/flash-style model since it will usually take too long otherwise.

## Saved pipeline runs

Every finished `@pipeline` run is saved under `$FORGER_CACHE/pipeline_runs`, one file per run, so variants never
overwrite each other. `runs()` prints the latest, oldest first: rows, cost, warnings, then the yields of steps that
dropped rows (95% intervals) and the mean last filter score.

```text
  main-20261022-130510-4242  13:05:10  2 steps → 20 rows  $0.60  LIMIT 50
      judge 0.40 (0.28–0.54) · mean filter.by_llm_score score 3.20 ± 0.20
```
