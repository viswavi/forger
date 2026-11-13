# `pyforger.filter`

Remove rows by criterion. If a filter empties the list, downstream ops receive `[]` and return `[]`. Signatures and
defaults are in [OPS.md](../OPS.md#filter).

| Filter | Reach for it when |
|---|---|
| `by_predicate` | Any Python check on a row; use it instead of a list comprehension. Not cached. |
| `by_llm_score` | Quality: an LLM judge scores each row on a rubric. See [LLM judges](#llm-judges). |
| `by_tests` | Code: run each row's Python or C++ program on its tests, keep rows passing `min_pass_rate`. |
| `by_rubric_score` | Your reward on every row: `grade(response, row)` scores each row, keep rows in bounds. |
| `by_pass_at_k` | Difficulty for *your* model: sample `k` answers, keep rows whose pass rate is in range. |
| `by_comparative_difficulty` | Difficulty: keep rows a strong model answers correctly and a weak model does not. |
| `by_judged_difficulty` | Cheap difficulty guess: a judge rates each task 1 to 5. |
| `program_verifier` | Not an op: a difficulty-filter `metric` that runs sampled programs on the row's tests. |
| `by_exact_match` | Dedup: keep up to `max_count` rows per unique field value. |
| `by_rouge_l` | Near-duplicates: drop rows whose ROUGE-L with any kept row exceeds `threshold`. |
| `by_embedding_similarity` | Semantic near-duplicates by embedding cosine; with `reference=`, also decontamination. |
| `by_n_gram_overlap` | Near-duplicates by word n-gram overlap with the kept rows. |
| `by_topic_frequency` | Topic balance: cluster rows, downsample over-represented topics, drop rare ones. |
| `by_exact_agreement` / `by_rouge_l_agreement` | Two fields of a row agree (or disagree, `disagree=True`). |
| `by_group_agreement` | Drop groups (by `group_by_input_field`) whose `match_output_field` values differ. |
| `by_length` | Word count of a field within bounds. |
| `by_invalid_response` | Drop empty responses or responses that copy the instruction. |
| `by_keyword_blocklist` | Drop responses with blocked phrases; the default list targets refusals. |
| `by_syntax` | Field parses as JSON, YAML, Python, or XML (optionally required keys or a JSON schema). |

**State what you expect a filter to keep** with `min_yield=` / `max_yield=` (output rows over input rows), accepted
by every alter, filter, and inspect op: `filter.by_llm_score(rows, ..., min_yield=0.3, max_yield=0.8)`. A violation
is a warning in the step report (`strict=True` raises); neither is part of the cache key.

**Choose thresholds from the score histogram.** The step report shows each filter's scores and the fraction of rows
nearby thresholds would keep (`by_rouge_l`, `by_n_gram_overlap`, `by_embedding_similarity`: current threshold
only). `by_llm_score` and `by_tests` also show up to three dropped rows with the reason (judge feedback, first failing
test).

`by_predicate` lists a Python check in the step report (`label` names it):

```python
rows = filter.by_predicate(rows, lambda r: len(r["papers"]) >= 3, label="has ≥3 papers", min_yield=0.5)
```

`by_rubric_score(rows, grade, *, response_field="response", min_score=None, max_score=None, score_field=None,
keep_ungradable=False)` scores every row with `grade(response, row)`, the function
[`inspect.rubric_scores`](inspect.md) takes. That report scores only a sample of `n` rows (default 50), so do not keep
or drop rows by it. Bounds are inclusive (None is open); `score_field` adds the score to kept rows. Ungradable rows (a
None response, a None grade, or a grade that raised) are dropped unless `keep_ungradable=True`; raised grades warn and
are not cached. Grades are cached apart from the bounds and shared with `rubric_scores`, so changing `min_score` makes
no `grade` calls. To score a model's own samples per row, use
`by_pass_at_k(..., metric=grade, extract_answer=lambda text: text)`.

```python
rows = filter.by_rubric_score(rows, grade, response_field="solution", min_score=1.0, score_field="rubric_score")
```

`by_tests` and `alter.run_programs` share their test format and options; see
[Running programs against tests](alter.md#running-programs-against-tests).

## Semantic dedup and decontamination

`by_embedding_similarity` is the semantic counterpart of `by_rouge_l`: walking the rows in input order, it drops a row
whose embedding cosine to a kept row is at least `threshold` (default 0.95, for near-verbatim copies of long problems;
use 0.9 for one-line rows, where one changed number scores 0.93). `reference=` (e.g. an eval set) drops every row that
close to any reference row; with a reference, `dedupe=True` (default) also dedups within the rows. The embedder is
chosen as in [`embedding_dispersion`](inspect.md) (`embedding_model`, default Qwen3-Embedding-0.6B, or `embed_fn`).
It drops the rows `embedding_dispersion` pairs as near-duplicates at `duplicate_threshold=threshold`, except chains:
a row near only an already dropped row is kept (A ≈ B ≈ C with A and C apart keeps A and C).

```python
rows = filter.by_embedding_similarity(rows, field="problem", threshold=0.9, reference=eval_rows,
                                      reference_field="question")
```

## LLM judges

`by_llm_score` rates each row from 1 to `max_score` against `criteria` and a rubric, feedback before score. Judges
tend to score nearly everything at the top:

- If the histogram is piled at the top, the threshold separates nothing: tighten `criteria` or the rubric.
- A criterion most outputs meet easily ("the scenario is realistic") scores nearly everything 5; one worded too
  literally ("same voice" for a rewrite that must drop sarcasm) marks down good outputs. Read a few rows' feedback
  (`explanation_field`, or the dropped rows in the step report) before trusting a threshold.
- Cost trap: a reasoning judge can cost more than the generation (GLM-5.3 at `reasoning_effort="low"`: about 7K
  output tokens, 92K token units, per call). Judge a few rows first and check the step report's units column.
- Feedback can be most of a judge call's output. `max_feedback_words=50` asks only for the mistakes that set the
  score, in at most 50 words; `0` asks for the score alone (`{"score": 3}`), leaving `explanation_field` and
  dropped-row notes empty. The default (None) asks for every mistake. It changes only the appended answer
  instruction: a template with its own `"score"` format keeps it.
- Judge scores are cached separately from `score_threshold` and `agreement_threshold`, so re-thresholding makes no
  new judge calls.
- `num_judges > 1` needs `temperature > 0`; `agreement_threshold` is compared with the fraction of judges scoring at
  least `score_threshold`.
- With several samples per input, `group_by="<input field>"` judges them side by side, which separates them best;
  each is still kept by the thresholds on its own.
- Scores keep fractions: `4.9` fails `score_threshold=5`.

The default rubric (`pyforger.filter.judge.DEFAULT_RUBRIC`) grades how serious the mistakes are (3: part is wrong;
4: minor shortcomings only), and `score_threshold=4` keeps 4 and 5. A custom `rubric` must describe exactly levels 1
to `max_score`; any `max_score` other than 5 needs one (and a `score_threshold` when it is below 4).

```python
kept = filter.by_llm_score(
    rows,
    model=judge,
    input_field="question",
    output_field="answer",
    criteria="the final answer is correct and every step is justified",
    rubric={
        1: "Wrong final answer.",
        2: "Right answer, but a step is wrong or missing.",
        3: "Right answer, every step correct, but some steps are asserted without justification.",
        4: "Right answer, every step correct and justified.",
    },
    max_score=4,
    score_threshold=4,
    explanation_field="judge_feedback",  # score_field="judge_score" adds the mean score
)
```

Failures and parsing (the first holds for every LLM filter):

- An unparseable answer, a request that still fails after retries (504, timeout), or a row error (content filter,
  prompt over the context window) is no vote, and the step report warns. After a transient failure the result is not
  cached, so a rerun asks again; a row error is cached like an answer, so a rerun gives the same votes. Errors a
  retry cannot fix (bad key, other 400s, spending cap, quota) raise.
- The no-score warning and the dropped rows' notes quote an unparseable reply, cut short
  (`an unparseable reply: "Score: 4 out of 5 …"`, or `(empty reply)` for a judge that spent `max_tokens` thinking).
- Answers are read after a closed `</think>`; a judge cut off inside `<think>` gives no score.
- `prompt_template` replaces the prompt; placeholders are the row's fields plus `{input}`, `{output}`,
  `{criteria}`, `{max_score}`, `{rubric}`. Write literal braces as `{{ }}`. The named placeholders win over row
  fields: a row's own `criteria` list never reaches `{criteria}` (the `criteria=` string), and a warning names each
  row field hidden that way.

## Difficulty filters

Each takes the task in `input_field` (text or chat messages). From cheapest and least reliable to most:

- `by_judged_difficulty`: one judge call per row guesses difficulty 1 to 5; keeps `[min_difficulty, max_difficulty]`
  (default `[3, 5]`). Edit the level 1, 3, and 5 descriptions (`*_difficulty_criterion`) for your domain. A row with
  no parseable rating is dropped; ratings are cached apart from the bounds.
- `by_comparative_difficulty`: one answer each from `weak_model` and `strong_model`; keeps rows only the strong
  model gets right.
- `by_pass_at_k`: `k` samples per row; keeps rows whose pass rate is in `[min_rate, max_rate]` (default
  `[0.1, 0.9]`). It is the only one that measures difficulty for the model you will train. With the student on your
  own GPUs (`LocalModel`, see [acquire.md](acquire.md)) it costs only GPU time. For code, pass
  `metric=filter.program_verifier("tests")` (`language="cpp"` for C++, `min_score=` for partial credit).
  A metric may return a score in [0, 1] instead of a bool (e.g. a partial-credit grader); the rate is then the
  row's mean score. `by_comparative_difficulty` needs pass/fail grades.
  `histogram=True` prints the pass-rate histogram outside a step and on a cache hit (not keyed), with a line
  counting rows whose samples all got one grade: no GRPO gradient there.
  The bounds can keep such rows under partial credit (samples all at 0.6 give rate 0.6); another line counts them
  among the kept rows, and a warning is logged when they are at least 20% of the kept rows (and at least 5).
  Under partial credit one more line histograms every sample's grade before filtering, with the highest and
  most common grade, since a rate is a mean and hides where samples land.

Grading guardrails for `by_comparative_difficulty` and `by_pass_at_k`:

- No answer format is imposed: the whole text after `</think>` is the answer. For `metric="exact_match"` ask for a
  bare answer or pass `extract_answer`, e.g. `lambda t: t.rsplit("ANSWER:", 1)[-1].strip()`.
- `metric` is `"exact_match"`, `"rouge_l"` (both need `ground_truth_field`), or a verifier
  `(prediction, row) -> bool | None` such as `filter.program_verifier("tests")`. Verifiers must be thread-safe.
- A `None` grade (empty answer not cut off, failed request, verifier returning `None`) is excluded, not failed.
  `by_pass_at_k` needs all `k` grades valid unless `min_valid` allows fewer.
- Truncated samples (`<think>` never closed, or an empty answer that stopped at `max_tokens`, e.g. a model returning
  its reasoning separately): `by_pass_at_k` grades them failed (`truncated="fail"`), `by_comparative_difficulty`
  excludes them. The step report counts both kinds (`34 of 47 samples truncated (stopped at max_tokens before any
  answer)`) and splits rows with no pass rate (`unparsed` in the histogram) into those that lost samples to
  truncation and those whose answers did not grade. If most are truncated, raise `max_tokens` or lower
  `reasoning_effort`.
- Predictions are cached apart from grades: changing the bounds or metric makes no model calls, and adding tasks or
  raising `k` samples only what is new.

`outcomes_field` stores each kept row's grades (`{"model": [...]}`, or `{"weak": [...], "strong": [...]}`), and
`completions_field` the raw completions in the same order, e.g. to reuse a student's passing samples as STaR targets:

```python
hard = filter.by_pass_at_k(
    rows,
    model=student,  # e.g. a LocalModel or TinkerModel; see acquire.md
    input_field="problem",
    ground_truth_field="final_answer",
    k=8,
    min_rate=0.125,
    max_rate=0.5,
    outcomes_field="evaluations",  # kept rows get e.g. {"model": [True, False, ...]}
    completions_field="samples",  # and {"model": ["<think>...</think>42", ...]}, in the same order
)
targets = [
    {**row, "response": text}
    for row in hard
    for text, ok in zip(row["samples"]["model"], row["evaluations"]["model"])
    if ok
]
```

**To measure pass@k without filtering** (e.g. how often a checkpoint already solves your prompts), pass
`min_rate=0.0, max_rate=1.0` with `outcomes_field`: every row with enough valid grades is kept, and the step report's
pass-rate histogram is the measurement.

`by_topic_frequency` clusters `field` with [`inspect.topic_based_clustering`](inspect.md), downsamples topics above
`max_frequency` of the rows (default twice the uniform share) and drops topics below `min_frequency`. The clustering
is cached on its own, so trying other bounds is free; pick them from the report's topic-share histogram. Use
`context` to say what topics should describe:

```python
balanced = filter.by_topic_frequency(
    rows,
    field="problem",
    model=model,
    context="competition mathematics problems. Topics should describe the technique, NOT the numeric answer.",
    num_clusters=20,
    max_frequency=0.1,  # no topic keeps more than 10% of the rows
)
```
