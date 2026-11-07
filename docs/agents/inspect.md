# `pyforger.inspect`

Inspect ops return a *report* about rows and never change them. Signatures are in [OPS.md](../OPS.md#inspect).

| Export | Cached | Use it to |
|---|---|---|
| `topic_based_clustering` | yes | Label each row with one of `num_clusters` LLM-named topics for one field. |
| `eval_tree` | yes | Build a capability tree and weakness profile from scored model outputs. |
| `inspect_step` | no | See more rows of a step: reprints its latest full report (10 rows); `rows=True` returns all. |
| `spend` | no | Print and return the spend ledger's totals by run, pipeline, and step. |
| `runs` | no | Print and return the saved `@pipeline` runs with step yields, e.g. `runs("main")`. |
| `compare_pipelines` | no | Print where two saved pipeline runs diverge and how each differing step's rows differ. |
| `phrase_frequency` | no | Print, per string field, the phrases and names the most rows share. No LLM. |
| `student_log_likelihood` | yes | How likely the student finds each row's target, in bits, beside its own responses. |
| `rubric_scores` | grades | Print and return a reward's score histogram on responses, samples, and controls. |
| `embedding_dispersion` | yes | Print and return how dispersed a field is in embedding space vs. a reference. No LLM. |
| `compare_to_reference` | yes | Print and return feature distributions (lengths, numbers, labels) beside a reference. |

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

`rubric_scores(rows, grade, *, n=50, response_fields="response", probes=None, model=None, k=4, ...)` bins a reward's
scores before RL. `grade(response, row)` returns a score in [0, 1] or None: a code rubric, an LLM judge, or exact match.
Columns: each response field (e.g. the SFT target), each probe, `k` samples per prompt from `model` (what RL sees), and
three controls no task should reward: `empty`, `random` (200 random characters), and `other_row` (another row's
response). On a rubric that gave 1.0 to any response without a python code block, the targets looked fine; `empty`
did not:

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
- `probes={"name": probe(row) -> response}`: responses a sound reward scores low, built per row: a program printing a
  valid but trivial answer, the example output hard-coded, the number from the statement. Flagged like the controls.
- The first field column (else the sampled one) is the reference; every other field, probe, and sampled column is
  compared with it row by row: `baseline: below solution on 31, tied 15, above 4 of 50 rows, mean gap +0.21`. Ties
  with a known-weak baseline are rows the reward cannot tell apart.
- `report.model_dump()` is plain JSON: bins, and per column counts, mean, shares at 0 and 1, per-row scores, and
  `vs_reference` (`below`, `tied`, `above`, `mean_gap`). Grades are cached (`grade` by its `cache_key()`).

## `embedding_dispersion`

`embedding_dispersion(rows, field, *, reference=None, k=5, duplicate_threshold=0.95, ...)` measures how spread out
`rows` are, ideally against a `reference` you want to resemble (e.g. retrieved documents or a seed dataset):

```text
  coverage 0.41  12/29 reference rows have a row within their 5-NN radius (29 random rows of each)
      not covered: "Find the remainder when 8^9 is divided by 7."  nearest 0.06
  off-distribution 0.07  2/29 rows are outside every reference row's 5-NN radius
  ! near reference 0.93  27/29 rows, on 24 reference rows (about 0.10 for rows drawn like the reference)
      3 rows near "A fair die is rolled 2 times. What is…"  closest 1.00: "A fair die is rolled 2 times. What is…"
  near-duplicates 0.17  5/29 rows in 4 pairs with cosine ≥ 0.95 (3 rows repeat an earlier row exactly)
      they form 2 groups, the largest of 3 rows: the 29 rows hold about 26 distinct ones
  Vendi 11.4  effective number of distinct rows among 29 (reference 17.5)
```

- *Coverage* ([Naeem et al. 2020](https://arxiv.org/abs/2002.09797)): share of reference rows with a row inside their
  k-NN ball; computed on equal-size samples (two samples of one distribution score about 0.97 at `k=5`).
- *Off-distribution*: share of rows outside every reference row's ball. *Vendi*: effective number of distinct rows.
- *Near reference*: share of rows more similar to their nearest reference row than 90% of reference rows are to
  theirs; about 0.10 for rows drawn like the reference, higher for rows that restate or vary individual reference
  rows (`!` at 0.25). Lists the reference rows they cluster on, each beside its closest row.
- *Near-duplicates*: 0.95 counts edited copies of long problems. With Qwen3-Embedding-0.6B, 220-word coding problems
  with whitespace, case, a few numbers, or one sentence changed score 0.95-1.00, and 1 pair among 1,131 distinct
  LiveCodeBench problems reaches 0.95 (5% of rows at 0.9). Use 0.9 for one-line rows: one changed number scores 0.93
  (median), the nearest different problem 0.66.
  [`filter.by_embedding_similarity`](filter.md) with the same threshold and embedding drops the later row of each pair.
  Rows linked by pairs, directly or through other rows, form a group: 40 instances of one template that differ only
  in numbers are one group of 40, so "415 rows hold about 13 distinct ones" says the set is 13 problems. `!` at 0.25.
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

`compare_to_reference(rows, reference, *, features=None, tokenizer=None, max_length=None)` prints each feature of
`rows` beside the same feature of `reference`, e.g. training rows beside validation rows: text by length (words, or
tokens with `tokenizer`), numbers by quantiles, labels by share. `!` marks a large gap. It makes no model calls. A
feature is a field, a `row -> value` function, or a `(rows_spec, reference_spec)` pair when the sets differ:

```python
inspect.compare_to_reference(rows, validation, features={
    "prompt": ("prompt", "task_text"),
    "abstains": (lambda r: r["ground_truth"].endswith(MARKER), "impossible"),
    "target": ("ground_truth", lambda v: None),  # rows only
}, max_length=4000)
```

```text
inspect compare_to_reference ───────────────────────────────────────────────────────────────────────────────────────────
  300 rows vs 137 reference rows

  feature            rows p10 / p50 / p90   reference p10 / p50 / p90   outside ref 5-95%   over 4,000
  ────────────────────────────────────────────────────────────────────────────────────────────────────────
  prompt (words)          117 / 249 / 365             131 / 257 / 439                 11%     0% vs 0%
  target (words)    2,387 / 4,058 / 5,620                           -                   -     52% vs -

  ! abstains: False 94% vs 63%, True 6% vs 37%  (distance 0.31)
```

## Saved pipeline runs

Every finished `@pipeline` run is saved under `$FORGER_CACHE/pipeline_runs`, one file per run, so variants never
overwrite each other. `runs()` prints the latest, oldest first: rows, cost, warnings, then the yields of steps that
dropped rows (95% intervals) and the mean last filter score.

```text
  main-20261022-130510-4242  13:05:10  2 steps → 20 rows  $0.60  LIMIT 50
      judge 0.40 (0.28–0.54) · mean filter.by_llm_score score 3.20 ± 0.20
```

## Comparing pipeline runs

Each saved run also records every step's cache key and a digest of each part of it (the step's code, each global it
reads, each argument). To try a variant, write it as a second pipeline in the same file that calls the same steps;
the steps they share hit the cache:

```python
@pipeline
def main_a(model):
    return verify(solve(find_problems(model), model, prompt=PROMPT_A))

@pipeline
def main_b(model):
    return verify(solve(find_problems(model), model, prompt=PROMPT_B))
```

After running both, `compare_pipelines("main_a", "main_b")` prints each step of both runs, its rows in → out and
cost, and why its cache keys differ:

```text
  step            A in → out     cost      B in → out     cost  difference
  find_problems      0 → 143    $0.00         0 → 143    $0.00  same
  solve            143 → 143    $0.41       143 → 143    $0.52  argument prompt differs
  verify            143 → 61    $0.04        143 → 88    $0.05  input rows differ (from an earlier step)
  first difference: solve

  verify:
    filter.by_tests: A 61/143 (0.43)  B 88/143 (0.62)  · B − A +0.19 (95% CI +0.07 to +0.30): clearly different
    rows (matched by id): 59 in both, 2 only in A, 29 only in B
      2 vs 29: clearly different (B keeps more of them; sign test p < 0.001)
      only in B: {"id": "p017", "statement": "Given n intervals…", ...}
```

Each difference comes with a verdict, `clearly different` or `within noise`: an op's yield difference with its 95%
interval, score means (`mean ... A 3.10 ± 0.20  B 3.60 ± 0.18`) compared by their intervals, and a sign test on the
rows only one run kept (`6 vs 4: within noise (... p = 0.75)`).

A step's difference is `code differs`, `global PROMPT differs`, `argument <name> differs`, or `input rows differ`
when only its rows argument changed (an earlier step caused it). For each step that differs it also lists, when both
outputs are still in the step cache, the output rows only one run has, matched by `key=` (default: an `id` field, or
another field unique in both runs), and for rows both runs have, which fields differ (`rows in both whose values
differ: solution in 120`). With no arguments it compares the latest run with the run before it (both limited or both
full); a name means that pipeline's latest run, and a run id from `runs()` names one run. Runs saved before steps
recorded their keys say `no key recorded (an older run)` for each step. It returns a dict: `first_difference`, and per
step `same`, `why`, `ops` (with `b_minus_a`: the difference, its `ci`, and `clear`), `scores`, and `rows` (with
`sign_test_p`).
