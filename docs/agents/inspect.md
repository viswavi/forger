# `pyforger.inspect`

Examine rows without modifying them: inspect ops return a *report* about the rows (e.g. `list[str]`), not rows. At any
point, inspection can be performed to generate a report about the current data, without altering the original rows or
producing new rows in the pipeline.

| Export | Cached | Description |
|---|---|---|
| `topic_based_clustering` | yes | Interpretable clustering of contents of one field from the input rows. |
| `eval_tree` | yes | Hierarchical capability tree and statistical weakness profile for scored outputs. |
| `inspect_step` | no | Print a `@step`'s latest full report, return its `StepRecord`; `rows=True`: its whole output. |
| `spend` | no | Print and return the spend ledger's totals by run, pipeline, and step. |
| `runs` | no | Print and return the saved `@pipeline` runs with step yields, e.g. `runs("main")`. |
| `phrase_frequency` | no | Print and return, per string field, the phrases and names the most rows share. No LLM. |
| `student_log_likelihood` | yes | How likely the student finds each row's target, in bits, beside its own responses. |
| `rubric_scores` | grades | Print and return a reward's score histogram on responses, samples, and controls. |
| `embedding_dispersion` | yes | Print and return how dispersed a field is in embedding space vs. a reference. No LLM. |

`phrase_frequency(rows, fields=None, *, ngram_range=(2, 4), top_k=10, min_share=0.05, max_rows=2000,
max_chars=2000)` counts document frequency: the share of rows whose value contains each word n-gram (lowercased) or
capitalized name, e.g. `rewrite  phrases: "i need" 78/79 (99%) · "i need you to" 46/79 (58%)` and `context  names:
"Maya" 46/79 (58%)`. It leaves out the same template text as the step report's `repeated:` line, lists a phrase only
when no longer phrase containing it is in nearly as many rows, and returns a `PhraseReport` (one `FieldPhrases` per
field, with `skipped` set for categorical, identifier, and code fields). Topic clustering can look balanced while
every row says "I need"; run both. Rows are not modified.

`student_log_likelihood(rows, model, *, prompt_field="prompt", target_field="response", own_samples=64,
max_tokens=None, top_k=5)` scores each row's target (e.g. an SFT target `"<think>...</think>answer"`) under the student,
a `LocalModel`, as SFT trains on it. Bits are `-log2` of each token's probability given the prompt and the earlier
tokens. A log-likelihood alone has no natural scale, so the student also answers `own_samples` of the prompts (evenly
spaced, at temperature 1) and its own responses are scored the same way:

```text
  134 rows · "ground_truth" scored by Qwen/Qwen3-4B
                  rows   median tokens   bits/token   bits/row   total bits
  targets          134             736         3.77      2,641   450.2k
    reasoning      134             600         4.18
    answer         134             126         0.45
  student's own    134           8,064         0.44      4,011   -
    reasoning      124           7,243         0.45
    answer         124             272         0.02

  excess over the student's own rate (0.44 bits/token): 397.8k bits in total
    per row: p10 290 · median 2,358 · p90 5,999 · max 12.8k
  most excess bits:
    row 68  12.8k (2,479 tokens × 5.58 bits)  ### Question: There is a row of N coins. In each turn, a player selects a…
```

Here (a coding SFT set for Qwen3-4B) the targets' reasoning costs 4.18 bits per token against the student's own 0.45:
a teacher's reasoning, far from what the student writes, though at a tenth of its length.

- *bits/token*, targets against the student's own: how foreign the text is, token by token.
- *tokens*: how long the targets are against what the student writes.
- *excess bits* (`RowLikelihood.excess_bits`): a target's bits minus what the student's own rate costs for the same
  number of tokens. About 0 for responses the student would write itself, such as STaR traces; the total grows with
  both foreignness and length, and is roughly how far SFT on the rows has to move the student. It describes the data;
  it has not been validated as a predictor of what SFT will do. Compare datasets, or rows within one, rather than
  reading one number alone.

The report lists the rows with the most excess bits; `report.rows[i]` belongs to `rows[i]`. Scores and the sampled
responses are cached, so changing `top_k` costs nothing. A `LocalModel` scores and samples on every visible GPU (its
default `data_parallel_size="auto"`). Scoring is fast; sampling the student's own responses is the slow part: 134
long reasoning responses took 37 minutes on one L40S and 19 on two. Its median rate is stable with the default 64
samples.

`rubric_scores(rows, grade, *, n=50, response_fields="response", model=None, k=4, prompt_field="prompt",
control_columns=True, seed=0, ...)` shows a reward's distribution before RL. `grade(response, row)` returns a score in
[0, 1] (a bool counts as 0 or 1) or None when ungradable: a code rubric, an LLM judge, or exact match. It grades `n`
rows drawn with `seed`, one column per response field (e.g. the SFT target), `k` samples per prompt from `model` (what
RL sees; the report counts prompts whose samples all got the same score, which give no gradient), and with
`control_columns` three responses no task should reward: `empty`, `random` (200 random characters), and `other_row`
(another row's response, never its own). A control earning half the reference column's mean is flagged. On a rubric
that gave 1.0 to any response without a python code block, the targets looked fine; `empty` did not:

```text
  50 of 60 rows (seed 0) · grade abstain_bug · sampled: toy-student

  score          response   sampled ×3   empty   random   other_row
  ─────────────────────────────────────────────────────────────────────
  0                    10           25       0        0          50
  (0, 0.25]             0            0       0        0           0
  ...
  1                    40          125      50       50           0
  ungradable            0            0       0        0           0
  mean               0.80         0.83    1.00     1.00        0.00

  sampled: 25/50 prompts gave all samples the same score (0 all 0, 25 all 1): no RL gradient there
  ! empty scores 1.00 on average, at least half of response's 0.80: RL can earn it too
  sampled ×3 highest: row 59 1.00  I cannot
```

It returns a `RubricScoreReport`; `report.model_dump()` is plain JSON data: `bins`, and per column `counts` (aligned
with `bins`), `mean`, `share_at_0`, `share_at_1`, `ungradable`, `errors`, the per-row `scores`, and the lowest and
highest examples. `grade` is part of the cache key (an object by its `cache_key()`), so the grades are cached; a
grading that raised is counted with its message and runs again next time. Rows are not modified.

`embedding_dispersion(rows, field, *, reference=None, reference_field=None, embed_fn=None, embedding_model=...,
k=5, duplicate_threshold=0.95, max_rows=2000, examples=3, seed=0)` measures how spread out `rows` are, ideally against
a `reference` of the rows you want to resemble (e.g. retrieved documents or a seed dataset). It returns a
`DispersionReport` (`report.model_dump()` gives a dict) and prints it. On 29 rows drawn from 2 of the 6 topics of a
96-row reference of math problems, plus 3 copies and 2 rows unlike the reference (`embedding_model="tfidf"`):

```text
inspect embedding_dispersion ───────────────────────────────────────────────────────────────────────────────────────────
  problem  29 rows · reference 96 rows · embedding tfidf
  coverage 0.41  12/29 reference rows have a row within their 5-NN radius (29 random rows of each)
      not covered: "Find the remainder when 8^9 is divided by 7."  nearest 0.06
      not covered: "How many positive divisors does 560 have?"  nearest 0.00
      not covered: "In how many ways can 9 people sit around a round table?"  nearest 0.02
  off-distribution 0.07  2/29 rows are outside every reference row's 5-NN radius
      outside: "Translate 'good morning' into French."  nearest 0.00
      outside: "Write a haiku about the ocean at dawn."  nearest 0.06
  near-duplicates 0.17  5/29 rows in 4 pairs with cosine ≥ 0.95 (3 rows repeat an earlier row exactly)
      1.00 "A fair die is rolled 2 times. What is the probability that a six appears at least once?"  exact copy
      1.00 "Two cards are drawn from a standard deck. What is the probability both are aces?"  exact copy
  Vendi 11.4  effective number of distinct rows among 29 (reference 17.5)
```

- **coverage** ([Naeem et al. 2020](https://arxiv.org/abs/2002.09797)): the share of reference rows with one of
  `rows` inside their k-NN ball (as close as their `k`-th nearest reference row). Low coverage means parts of the
  reference are missing; `not covered:` lists them. It grows with the number of rows, so it is computed on equal-size
  random samples of both sets. Two samples of one distribution score about 0.97 at `k=5`.
- **off-distribution**: the share of `rows` outside every reference row's ball, e.g. off-topic or malformed rows.
- **near-duplicates**: rows in a pair with cosine similarity of at least `duplicate_threshold`. 0.95 counts
  near-verbatim copies. With Qwen3-Embedding-0.6B, copies of 220-word coding problems with whitespace, case, a few
  numbers, or one sentence changed score 0.95 to 1.00 (at least 92% of each kind), and among 1,131 distinct
  LiveCodeBench problems one pair reaches 0.95 ("Maximum Glutton" and "Minimum Glutton"); at 0.9, 5% of those rows
  would be listed. One-line math problems score lower: one changed number 0.93 (median), a paraphrase 0.88, a
  template variant (same structure, other numbers and objects) 0.77, the nearest different problem 0.66. Lower it to
  0.9 for short rows. Under TF-IDF, one changed number in a one-line problem scores 0.83 and a paraphrase 0.20.
  [`filter.by_embedding_similarity`](filter.md) with the same threshold and embedding drops the later row of each pair.
- **Vendi** ([Friedman & Dieng 2023](https://arxiv.org/abs/2210.02410)): the effective number of distinct rows, from 1
  (all the same) to the sample size, on the same equal-size samples. It cannot exceed the embedding's dimension (at
  most 256 for TF-IDF).

Without a `reference`, only near-duplicates and Vendi are reported. At most `max_rows` rows of each set are embedded
(a random sample with `seed`); both sets are embedded in one call, with the same backends as `topic_based_clustering`
below (`embed_fn`, `embedding_model="tfidf"`, or a sentence-transformers model, falling back to TF-IDF). The numbers
depend on the embedding, so compare datasets (or a dataset against its reference) under the same embedding rather
than reading one number alone.

Topic-based clustering uses the ClusterFusion clustering algorithm ([arXiv:2512.04350](https://arxiv.org/abs/2512.04350)),
a hybrid method in which embeddings guide how rows are partitioned and an LLM summarizes the partitions into
interpretable topics, then assigns each row to one of them.

```python
from pyforger.inspect import topic_based_clustering

clusters = topic_based_clustering(
    rows,
    field="problem",
    model=model,
    context="competition mathematics problems. Topics should describe the "
            "mathematical subject and technique, NOT the numeric answer.",
    num_clusters=20,
)
# clusters[i] is the topic name of rows[i]
```

`context` is an optional hint that describes the corpus and what the topics should capture; it improves topic names.
Without it, the prompts describe the corpus as the field's values (e.g. "'problem' values from a dataset").

`filter.by_topic_frequency` runs this clustering (cached) to downsample over-represented topics.

With fewer unique `field` values than `num_clusters` (e.g. 6 for `num_clusters=20`), the op uses one cluster per
unique value and logs a warning, so there is no need to shrink `num_clusters` for small inputs yourself. With a single
unique value, every row gets that value (cut to 80 characters) as its topic, without calling the LLM.

Embeddings only decide which rows the LLM summarizes together. By default they come from the sentence-transformers model
`embedding_model` (`pip install "pyforger[clustering]"`, GPU when available; n values embed on `min(n // 500 + 1,
visible GPUs)` GPUs, e.g. 400 on one and 999 on two; batches hold up to `FORGER_EMBED_BATCH_TOKENS` padded tokens,
default 16384; a longer text is a batch of its own). `embedding_model="tfidf"` uses a built-in TF-IDF + SVD embedder
on the CPU (scikit-learn only); if sentence-transformers cannot be imported, the op falls back to it with a warning.
To use another backend, pass
`embed_fn`: a function that takes the field values as a `list[str]` and returns an array-like of shape `(n_rows, dim)`,
one vector per string. `embed_fn` overrides `embedding_model`. The fallback is resolved before the cache lookup, so its
result is cached as `"tfidf"` and is not reused once sentence-transformers is installed.

EvalTree ([paper](https://arxiv.org/abs/2503.08893)) annotates each problem with a required capability,
embeds those annotations, recursively groups them using K-means and cosine silhouette scores, and describes
each capability subtree. For binary correctness scores, each node reports accuracy and an exact binomial
confidence interval; subtrees whose interval is entirely below `threshold` form a weakness profile. The
implementation is adapted from the [reference repository](https://github.com/zhiyuan-zeng/EvalTree).

```python
import json
from pyforger.inspect import eval_tree

with open("val_predictions.jsonl") as f:  # one scored model output per row
    rows = [json.loads(line) for line in f]

model = Model("litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731", reasoning_effort="none")
report = eval_tree(rows, input_field="question", score_field="score", model=model,
                   threshold=0.5)
# report["tree"] has descriptions, row indices, accuracy, and child nodes.
# report["weaknesses"] lists the capability groups significantly below threshold.
# report["lowest"] lists the five subtrees with the lowest accuracy bounds.
```

With a few hundred rows a subtree's interval is rarely entirely below `threshold`, so `weaknesses` is often
empty (the op warns); read `lowest` then. Each annotation is one phrase of at most 300 characters; answers that are
not (empty after running out of tokens, or code) are asked again, and rows that still get none are left out of the
tree and listed in `report["unannotated"]`.

Inspect ops make one short LLM call per row (or per cluster). Run `eval_tree` with `reasoning_effort="none"`: on
137 coding problems, `"low"` cost 15 times as much and 10% of its annotations ran out of tokens in reasoning
loops. `topic_based_clustering` works with `"low"` or `"none"`; higher efforts make a few calls reason for
minutes and hit gateway timeouts. Both ops warn about it.

The `question` alone is used to infer the skill, avoiding leakage from an incorrect model output. If verified
solutions are available, pass `reference_field` to include them in the capability annotation prompt. The
default embedding backend is the same as `topic_based_clustering` (`pyforger[clustering]`); `embed_fn` can
provide another backend. `threshold`, `alpha`, `min_ci_size`, `min_child_size`, `max_children`, and `seed`
control weakness extraction and tree construction. The result contains only ordinary Python dicts and lists
and is JSON serializable.

## Saved pipeline runs

Every finished `@pipeline` run is saved under `$FORGER_CACHE/pipeline_runs`, one file per run, so two variants of a
pipeline never overwrite each other's record. `runs(name=None, last=10)` prints the latest runs (of pipeline `name`
if given), two lines per run, oldest first: its id, time, rows, cost, and warnings, then the yields of steps that
dropped rows (with 95% intervals) and the mean of the last filter score. It returns the `PipelineRecord`s.

```text
  main-20261022-130510-4242  13:05:10  2 steps → 20 rows  $0.60  LIMIT 50
      judge 0.40 (0.28–0.54) · mean filter.by_llm_score score 3.20 ± 0.20
```
