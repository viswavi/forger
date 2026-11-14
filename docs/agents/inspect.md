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
| `compare_pipelines` | no | Print where two saved pipeline runs diverge and how each differing step's rows differ. |
| `phrase_frequency` | no | Print and return, per string field, the phrases and names the most rows share. No LLM. |
| `student_log_likelihood` | yes | How likely the student finds each row's target, in bits, beside its own responses. |
| `rubric_scores` | grades | Print and return a reward's score histogram on a sample of `n` rows, beside controls. |
| `rubric_sensitivity` | rewrites | Compare paraphrases' scores and measure identical-text grading noise. |
| `rubric_pair_accuracy` | grades | How often a reward scores the better of your ordered pairs higher, per criterion. |
| `embedding_dispersion` | yes | Print and return how dispersed a field is in embedding space vs. a reference. No LLM. |
| `compare_to_reference` | yes | Print and return feature distributions (lengths, numbers, labels) beside a reference. |

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

`rubric_scores(rows, grade, *, n=50, response_fields="response", reference_field=None, probes=None, model=None, k=4,
teacher=None, teacher_k=2, rubric_field="rubric", prompt_field="prompt", control_columns=True, seed=0, ...)` shows a
reward's
distribution before RL. `grade(response, row)` returns a score in [0, 1] (a bool counts as 0 or 1) or None when
ungradable: a code rubric, an LLM judge, or exact match. It scores a sample, not every row: `n` rows (default 50) drawn
with `seed`. The report's first line says how many (`50 of 300 rows sampled (seed 0; the other 250 have no score)`), and
an `n` below the row count logs a warning. `n=None` scores every row; to keep or drop rows by score, use
[`filter.by_rubric_score`](filter.md), which scores every row with the same `grade` and shares its cached grades. It
grades one column per response field (e.g. the SFT target), `k` samples per prompt from `model` (what RL sees; the
report counts prompts whose samples all got the same score, which give no gradient), and with `control_columns` three
responses no task should reward: `empty`, `random` (200 random characters), and `other_row` (another row's response,
never its own). A control earning half the reference column's mean is flagged. On a rubric that gave 1.0 to any response
without a python code block, the targets looked fine; `empty` did not:

```text
  50 of 60 rows sampled (seed 0; the other 10 have no score) · grade abstain_bug · sampled: toy-student

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

Two more checks ask whether the reward separates good responses from bad ones, not only whether it is computed
right. `probes={"name": probe(row) -> response}` adds a column of responses built per row that a sound rubric should
score low: for a program-checked task, e.g. one printing a valid but trivial answer, or the example output
hard-coded; for an answer-checked one, the number from the problem statement. A probe earning half the reference's
mean is flagged like a control. And the first field column (else the sampled one) is the reference: every other field,
probe, and sampled column is compared with it row by row, e.g. with `response_fields=["solution", "baseline"]`:

```text
  baseline: below solution on 31, tied 15, above 4 of 50 rows, mean gap +0.21
  trivial: below solution on 50, tied 0, above 0 of 50 rows, mean gap +0.88
```

Ties between a reference and a known-weak baseline are rows where the rubric cannot tell them apart; a sampled column
tied on most rows is a student that already earns the reference's score. A sampled column whose mean is 0.9 or more is
flagged: little is left for RL to raise.

A reference that scores well on its own rubric does not show that the rubric pays good replies: a rubric written from
the reference, or from the notes the reference was written from, can demand the reference's own details. `teacher=`
adds a column of `teacher_k` replies per prompt (default 2) from a model trusted to answer well, which sees only the
prompt. When rows carry a rubric in `rubric_field` (default `"rubric"`, as `[[criterion, weight], ...]`), each
criterion with a nonzero weight is also scored alone, on the reference and on every teacher reply (`grade` gets the row
with `[[criterion, 1]]` in that field), and passes at 0.5. A negative-weight criterion names a mistake and passes below
0.5, e.g. `["Recommends aspirin for a child", -75]` passes on a reply scored 0 on it. The report counts the criteria
the reference passes and every teacher reply fails, flags 20% or more, and prints the five heaviest (largest absolute
weight). The reference is an answer already stored in the rows: the first of `response_fields`, or the field
`reference_field` names (e.g. `"ground_truth"`). The teacher is a model writing fresh replies:

```text
  per criterion: teacher ×2 fails 31 of the 120 criteria reference passes on every reply, 18 on some (152 criteria on 40 rows)
  ! teacher fails 31 of the 120 criteria reference passes (26%) on every reply: read them below
  teacher fails: row 12 criterion 1 [100] (reference 1.00; teacher 0.00, 0.00)  Mentions that the starting dose is 81 mg…
```

Read the listed criteria. One may demand a detail, number or choice that a good reply can leave out, a mistake may be
docked on a reply that names it only to warn against it, or the teacher may be wrong; the count says where to look, not
which criteria are bad. Two replies per prompt matter: two samples of one model often disagree on a criterion, so one
reply's miss is weak evidence. The stronger the teacher, the stronger the check, and a model that did not write the
rubric or the reference does not share their habits. The check calls `grade` `1 + teacher_k` times per criterion and
costs `teacher_k` generations per sampled row.

It returns a `RubricScoreReport`; `report.model_dump()` is plain JSON data: `bins`, and per column `counts` (aligned
with `bins`), `mean`, `share_at_0`, `share_at_1`, `ungradable`, `errors`, the per-row `scores` (`scores[i]` belongs to row
`report.rows[i]`, the sampled rows only), `vs_reference` (`below`, `tied`, `above`, `mean_gap`), the lowest and highest
examples, and, with a teacher and rubrics, `teacher_criteria` (`criteria`, `reference_passes`, `fails_all`,
`fails_some`, `share_fails_all`, `failed`: every criterion that fails every teacher reply, heaviest first, with its
`row` and `index`; `examples` is its first five). `grade` is part of the cache key (an object by its `cache_key()`), so the
grades are cached; a grading that raised is counted with its message and runs again next
time. Rows are not modified.

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
  ! near reference 0.93  27/29 rows, on 24 reference rows (about 0.10 for rows drawn like the reference)
      counts rows more similar to their nearest reference row than 90% of reference rows are to theirs (0.85)
      nearest reference p50 1.00 / p90 1.00 · reference to nearest other p50 0.79 / p90 0.85 · 27 at cosine ≥ 0.95
      3 rows near "A fair die is rolled 2 times. What is…"  closest 1.00: "A fair die is rolled 2 times. What is…"
      2 rows near "Two cards are drawn from a standard d…"  closest 1.00: "Two cards are drawn from a standard d…"
      1 row near "Solve for x: 7x + 8 = 10x - 4."  closest 1.00: "Solve for x: 7x + 8 = 10x - 4."
  near-duplicates 0.17  5/29 rows in 4 pairs with cosine ≥ 0.95 (3 rows repeat an earlier row exactly)
      they form 2 groups, the largest of 3 rows: the 29 rows hold about 26 distinct ones
      1.00 "A fair die is rolled 2 times. What is the probability that a six appears at least once?"  exact copy
      1.00 "Two cards are drawn from a standard deck. What is the probability both are aces?"  exact copy
  Vendi 11.4  effective number of distinct rows among 29 (reference 17.5)
```

- **coverage** ([Naeem et al. 2020](https://arxiv.org/abs/2002.09797)): the share of reference rows with one of
  `rows` inside their k-NN ball (as close as their `k`-th nearest reference row). Low coverage means parts of the
  reference are missing; `not covered:` lists them. It grows with the number of rows, so it is computed on equal-size
  random samples of both sets. Two samples of one distribution score about 0.97 at `k=5`.
- **off-distribution**: the share of `rows` outside every reference row's ball, e.g. off-topic or malformed rows.
- **near reference**: the share of `rows` more similar to their nearest reference row than 90% of reference rows are
  to their nearest other one. Rows drawn like the reference score about 0.10; rows that restate, re-parameterize, or
  vary individual reference rows score higher, even below `duplicate_threshold`, and `!` marks 0.25 or more. The
  lines below list the reference rows those rows cluster on, each with its closest row, and the count at cosine of
  at least `duplicate_threshold` to a reference row. Here the rows copy reference problems, so 27 of 29 count.
- **near-duplicates**: rows in a pair with cosine similarity of at least `duplicate_threshold`. 0.95 counts
  near-verbatim copies. With Qwen3-Embedding-0.6B, copies of 220-word coding problems with whitespace, case, a few
  numbers, or one sentence changed score 0.95 to 1.00 (at least 92% of each kind), and among 1,131 distinct
  LiveCodeBench problems one pair reaches 0.95 ("Maximum Glutton" and "Minimum Glutton"); at 0.9, 5% of those rows
  would be listed. One-line math problems score lower: one changed number 0.93 (median), a paraphrase 0.88, a
  template variant (same structure, other numbers and objects) 0.77, the nearest different problem 0.66. Lower it to
  0.9 for short rows. Under TF-IDF, one changed number in a one-line problem scores 0.83 and a paraphrase 0.20.
  [`filter.by_embedding_similarity`](filter.md) with the same threshold and embedding drops the later row of each pair.
  Rows linked by pairs, directly or through other rows, form a group (`near_duplicate_groups`, `largest_group`):
  40 instances of one template that differ only in numbers are one group of 40, so "415 rows hold about 13 distinct
  ones" says the set is 13 problems. `!` marks a share of 0.25 or more.
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

Every finished `@pipeline` run is saved under `$FORGER_CACHE/pipeline_runs`, one file per run, so two variants of a
pipeline never overwrite each other's record. `runs(name=None, last=10)` prints the latest runs (of pipeline `name`
if given), two lines per run, oldest first: its id, time, rows, cost, and warnings, then the yields of steps that
dropped rows (with 95% intervals) and the mean of the last filter score. It returns the `PipelineRecord`s.

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

## Rubric sensitivity

`rubric_sensitivity(rows, grade, model=rewriter, n=50, k=2, repeats=2, epsilon=0.05)` measures how a reward changes
under candidate paraphrases. It uses `alter.paraphrase`, which sees the task and response, never the
rubric. Supply `variants_field="paraphrases"` instead of `model` to inspect existing rewrites (including manually
reviewed ones). Rows are sampled with `seed`; `n=None` inspects all rows. The original rows are unchanged.

```python
report = inspect.rubric_sensitivity(
    rows, grade, model=rewriter, response_field="solution", n=30, k=2,
    repeats=2, epsilon=0.05,
)
# report.pairs keeps the full original/rewrite texts and all repeated scores.
```

The report compares repeated-score means and counts absolute changes larger than `epsilon`. It separately reports
score ranges from grading the *identical text* repeatedly: judge randomness may already cause disagreement.
Grading bypasses Forger caches, including nested ops; disable any separate cache inside your grader too.
Paraphrase generation is cached separately. Each call grades fresh, at most `n * (k + 1) * repeats` times, plus
`n * k` generated rewrites. Each grade may itself execute many rubric criteria. At `repeats=1`, repeat noise is
unmeasured. This comparison does not establish statistical significance or subtract away judge noise.
Full-rubric and isolated criterion grading use independent draws; judge noise can make their results differ.

`equivalent(original, candidate, row)` optionally returns `True`, `False`, or `None` from an independent review.
Rejected pairs remain available but are excluded from summary comparisons. `candidate_tie_rate` includes unverified
pairs; `verified_tie_rate` includes only accepted pairs, so its reliability depends on that review. Without a check,
the report measures *candidate* sensitivity: a rewrite can accidentally fix an error, weaken a claim, or violate an
exact wording constraint. Such changes can legitimately change reward. Constant rewards pass this inspection too;
use `rubric_scores` controls and teacher checks to assess discrimination. Paraphrases probe a subset of equally
useful responses, not every equivalence class discussed in [Discretizing Reward Models](https://arxiv.org/abs/2606.21795).

Unchanged rewrites, duplicates, missing completions, rejected pairs, and ungradable pairs are counted separately.
No comparisons gives `None` for the tie rate, never perfect stability. The largest differences are previewed; use
`report.model_dump()` for all pairs and scores. This inspection reports evidence and does not filter or rewrite rubrics.

### Three sensitivity rates and criterion exploration

The report exposes `ChangeRate` objects with `changed`, `compared`, and `fraction` (multiply by 100 for percent):

| Field | Numerator | Denominator |
|---|---|---|
| `rubric_pair_changes` | Full-rubric original/paraphrase pairs changing beyond `epsilon` | Eligible pairs |
| `rubric_changes` | Input rows with at least one changing pair | Rows with at least one eligible pair |
| `criterion_changes` | Criterion occurrences with at least one changing pair | Criteria with an eligible pair |

A criterion occurrence is `(row, index)`, so identical wording in two tasks is counted twice. Each rate is also
available with a `verified_` prefix, using only pairs accepted by the optional equivalence review. Rejected or
ungradable comparisons never count as stable; a zero denominator gives `fraction=None`. With two rewrites each
for ten rubrics, two changes on separate rubrics mean 2/20 pairs (10%) but 2/10 rubrics (20%).

`per_criterion=True` (default) scores each nonzero-weight criterion separately from `rubric_field="rubric"`.
It calls the supplied grade with that field replaced by `[[criterion, 1.0]]`, so **the grade must read the supplied
rubric**. Use `per_criterion=False` for an opaque reward that ignores it. Negative weights are kept in the report,
but isolated scores use +1 so clipping cannot hide changes. `criterion_epsilon` defaults to `epsilon`.
Missing rubrics and zero-weight criteria are reported separately. Each criterion costs an additional
`(k + 1) * repeats` grade calls. All three rates measure observed changes, not statistical proof of oversensitivity.

```python
for entry in report.sensitive_criteria(verified_only=True):
    print(entry.row, entry.index, entry.criterion, entry.weight)
    for pair in entry.pairs:
        print(pair.original_scores, pair.candidate_scores, pair.equivalent)
        # pair.original and pair.candidate contain the full responses.
suspect = {(entry.row, entry.index) for entry in report.sensitive_criteria()}
# These indices address the original input rows, even when n sampled a subset.
```

`report.criteria` includes every tested criterion, including stable and ungradable ones, with raw paired scores,
repeat-noise ranges and errors. `sensitive_criteria()` ranks candidates by their largest eligible absolute gap.
The `(row, index)` set supports downstream selection or filtering, but does not automatically delete criteria:
review meaning preservation and judge noise first, and ensure any edited rubric still has a positive weight.

## Rubric pair accuracy

`rubric_pair_accuracy(rows, grade, *, better_field="better", worse_field="worse", n=None, repeats=1,
per_criterion=True, rubric_field="rubric")` asks whether a reward ranks responses the way you do. Each row holds one
text or a list of texts in each field, and every (better, worse) combination on the row is a pair. What makes one
better is yours to decide, e.g. a reference against a copy with a key fact removed or a mistake added, or a strong
model's reply against a weak one's. Each pair is a win, tie or loss with a margin, mean(better) - mean(worse):

```python
report = inspect.rubric_pair_accuracy(rows, grade=rubric_score(judge=True), repeats=2)
# Rubric pair accuracy: 40 of 40 rows; 80 pairs compared, 0 ungradable, 0 missing a text.
# Better text scored higher in 52, tied in 21, lower in 7 (accuracy 78.1%, ties count half; mean margin +0.214).
report.losing_pairs()        # pairs the worse text won, with their row
report.wrong_way_criteria()  # criteria that passed a worse text and failed its better one, with (row, index)
```

Ties are counted apart because rubrics with yes/no criteria tie often. With `per_criterion`, each nonzero-weight
criterion is scored alone on every text (`grade` gets the row with `[[criterion, 1]]` in `rubric_field`); a criterion
is right on a pair when it passes the better text and fails the worse, wrong for the reverse, and a mistake (negative
weight) passes a text that does not make it. `repeats > 1` grades each text again with fresh grades and records each
text's range in `identical_response_gaps`: a margin inside that range may be judge noise. It calls `grade` `repeats`
times per distinct text, and as often again per criterion. Nothing is removed or changed.
