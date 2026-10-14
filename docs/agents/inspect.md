# `pyforger.inspect`

Examine rows without modifying them: inspect ops return a *report* about the rows (e.g. `list[str]`), not rows. At any
point, inspection can be performed to generate a report about the current data, without altering the original rows or
producing new rows in the pipeline.

| Export | Cached | Description |
|---|---|---|
| `topic_based_clustering` | yes | Interpretable clustering of contents of one field from the input rows. |
| `eval_tree` | yes | Hierarchical capability tree and statistical weakness profile for scored outputs. |

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

report = eval_tree(rows, input_field="question", score_field="score", model=model,
                   threshold=0.5)
# report["tree"] has descriptions, row indices, accuracy, and child nodes.
# report["weaknesses"] lists the extracted capability groups.
```

Inspect ops make one short LLM call per row (or per cluster), so run them with `reasoning_effort="low"` or
`"none"`; higher efforts make a few calls reason for minutes and hit gateway timeouts, and both ops warn about it.

The `question` alone is used to infer the skill, avoiding leakage from an incorrect model output. If verified
solutions are available, pass `reference_field` to include them in the capability annotation prompt. The
default embedding backend is the same as `topic_based_clustering` (`pyforger[clustering]`); `embed_fn` can
provide another backend. `threshold`, `alpha`, `min_ci_size`, `min_child_size`, `max_children`, and `seed`
control weakness extraction and tree construction. The result contains only ordinary Python dicts and lists
and is JSON serializable.
