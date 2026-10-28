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
- Sampling the student's own responses is the slow part (134 long responses: 37 minutes on one L40S); cached.

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
