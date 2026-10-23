# `pyforger.inspect`

Examine rows without modifying them: inspect ops return a *report* about the rows (e.g. `list[str]`), not rows. At any
point, inspection can be performed to generate a report about the current data, without altering the original rows or
producing new rows in the pipeline.

| Export | Cached | Description |
|---|---|---|
| `topic_based_clustering` | yes | Interpretable clustering of contents of one field from the input rows. |
| `eval_tree` | yes | Hierarchical capability tree and statistical weakness profile for scored outputs. |
| `inspect_step` | no | Print the full report of the latest run of a `@step` and return its `StepRecord`. |
| `spend` | no | Print and return the spend ledger's totals by run, pipeline, and step. |
| `runs` | no | Print and return the saved `@pipeline` runs with step yields, e.g. `runs("main")`. |
| `phrase_frequency` | no | Print and return, per string field, the phrases and names the most rows share. No LLM. |

`phrase_frequency(rows, fields=None, *, ngram_range=(2, 4), top_k=10, min_share=0.05, max_rows=2000,
max_chars=2000)` counts document frequency: the share of rows whose value contains each word n-gram (lowercased) or
capitalized name, e.g. `rewrite  phrases: "i need" 78/79 (99%) · "i need you to" 46/79 (58%)` and `context  names:
"Maya" 46/79 (58%)`. It leaves out the same template text as the step report's `repeated:` line, lists a phrase only
when no longer phrase containing it is in nearly as many rows, and returns a `PhraseReport` (one `FieldPhrases` per
field, with `skipped` set for categorical, identifier, and code fields). Topic clustering can look balanced while
every row says "I need"; run both. Rows are not modified.

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

Embeddings only decide which rows the LLM summarizes together. By default they come from the sentence-transformers
model `embedding_model` (`pip install "pyforger[clustering]"`, GPU when available). `embedding_model="tfidf"` uses a
built-in TF-IDF + SVD embedder on the CPU (scikit-learn only); if sentence-transformers cannot be imported, the op
falls back to it with a warning. To use another backend, pass `embed_fn`: a function that takes the field values as a
`list[str]` and returns an array-like of shape `(n_rows, dim)`, one vector per string. `embed_fn` overrides
`embedding_model`. The fallback is resolved before the cache lookup, so its result is cached as `"tfidf"` and is not
reused once sentence-transformers is installed.

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
