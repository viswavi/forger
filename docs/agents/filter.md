# `pyforger.filter`

Remove rows by criterion. If a filter empties the list, downstream operations receive `[]` and return `[]`.

| Export | Description |
|---|---|
| `by_exact_agreement` | Keep rows where two fields agree (or disagree, with `disagree=True`). |
| `by_rouge_l_agreement` | ROUGE-L similarity gate between two fields. |
| `by_group_agreement` | Drop groups of rows (by `group_by_input_field`) whose `match_output_field` values differ. |
| `by_exact_match` | Keep up to `max_count` rows per unique field value. |
| `by_rouge_l` | Pairwise ROUGE-L diversity filter. |
| `by_n_gram_overlap` | N-gram overlap diversity filter. |
| `by_length` | Filter by word count within a given field. |
| `by_invalid_response` | Drop empty, identical, or near-identical responses to the instruction. |
| `by_keyword_blocklist` | Drop responses with certain keywords; default blocklist targets common refusal phrases. |
| `by_llm_score` | LLM judge scoring on a rubric, feedback before score; multi-judge consensus, `group_by` candidates. |
| `by_judged_difficulty` | LLM judge rates each task's difficulty from 1 to 5; keeps rows within a range. |
| `by_syntax` | Validate JSON, YAML, Python, or XML, optionally against required keys or a JSON schema. |
| `by_predicate` | Keep rows where `fn(row)` is truthy: a plain Python check that shows in the step report. Not cached. |
| `by_tests` | Run each row's stdin/stdout program on its tests; keep rows passing at least `min_pass_rate`. |
| `by_comparative_difficulty` | Keep rows a strong model answers correctly and a weak model does not. |
| `by_pass_at_k` | Sample `k` answers per row; keep rows whose pass rate is within `[min_rate, max_rate]`. |
| `program_verifier` | Not an op: a `metric` for the difficulty filters that runs sampled programs on the row's tests. |
| `by_topic_frequency` | Cluster rows into topics; downsample over-represented topics and drop rare ones. |

These are important for assuring quality and diversity in generated data, and one should experiment with their parameters
and thresholds to find the right balance for a given use case. For example, `by_rouge_l` with a very low threshold
produces more diverse outputs but may filter out many rows; `by_llm_score` with multiple judges and a high consensus
threshold produces higher-quality outputs but at the cost of more LLM calls.

`by_rouge_l` and `by_rouge_l_agreement` give the same scores as `rouge_score`'s `RougeScorer(["rougeL"])` without
stemming: text is lowercased and split at every character outside `[a-z0-9]` (so accented and non-Latin letters are
separators, not words), and the score is the F1 of the longest common subsequence of tokens. `by_rouge_l` compares
each row with every kept row, so its time grows with rows times kept rows; it scores a row against the whole kept pool
in one bit-parallel pass, e.g. 157 messages of 50–600 words take about 0.2s (2m16s with `rouge_score`).

Every op that takes input rows (alter, filter, inspect) accepts `min_yield=` and `max_yield=`: the expected range of
output rows over input rows, e.g. `filter.by_llm_score(rows, ..., min_yield=0.3, max_yield=0.8)`. A violation, or an
empty input with `min_yield`, is a warning in the step report (`strict=True` raises); neither keyword is part of the
cache key. `acquire.*` ops reject them. Filters also record a score per row, so the step report shows a histogram and
the fraction of rows each nearby threshold would keep; use it to choose thresholds. `by_rouge_l` and
`by_n_gram_overlap` are the exception: each score compares a row with the rows kept so far, so another threshold would
change the scores, and the report shows only the current threshold. `by_llm_score` and `by_tests` also note why each
row scored as it did (the judge's feedback, the first failing test), and under the histogram the report shows up to
three dropped rows with their note. With `num_judges > 1`, `by_llm_score` records the
fraction of judges scoring at least `score_threshold`, which is what `agreement_threshold` is compared with.
`by_llm_score` caches the judge scores separately from its thresholds, so re-running it with a different
`score_threshold` or `agreement_threshold` makes no new judge calls (`_no_cache=True` re-asks the judges;
`_cache_only=True` finds only thresholds that were run before).

Write a Python check on rows as `by_predicate` rather than a list comprehension, so the step report lists it with
rows in/out and a kept/dropped bar (`label` names it there; default: the function's name, or `predicate` for a
lambda). An exception from `fn` propagates with the row index in its message. It counts as neither a quality nor a
diversity filter.

```python
rows = filter.by_predicate(rows, lambda r: len(r["papers"]) >= 3, label="has ≥3 papers", min_yield=0.5)
```

## LLM judges

`by_llm_score` rates each row on 1 to `max_score`. Absolute judges tend to score nearly everything at the top of the
scale. Two things in the default prompt counter that, following Prometheus
([Kim et al., 2024](https://arxiv.org/abs/2310.08491)): a rubric that describes every score level, not just the
endpoints, and feedback written before the score, like chain of thought. The judge checks the output against each part
of `criteria` and lists every mistake, quoting the part it concerns and naming the highest score the rubric allows with
it (or, if it finds none, says what it checked); the score is at most the lowest of those caps, as
`{"feedback": "...", "score": 3}`. The prompt fences the input and output in `<input>` / `<output>` tags and says that
length or detail the criteria do not ask for counts for nothing. The feedback costs more output tokens per judge call.
Still check the step report's histogram; if it is piled up at the top, the threshold is not separating anything
(tighten `criteria` or the rubric). The criteria matter as much as the prompt: a criterion most outputs meet easily
("the scenario is realistic") still scores nearly everything 5, and one worded too literally ("same voice" for a
rewrite that must drop sarcasm) marks down good outputs, so read a few rows' feedback before trusting a threshold.
The step report helps: under the histogram it shows up to three dropped rows (their `output_field`, cut to 200
characters) with the judge's feedback, the lowest-scoring judge's when there are several.
With several samples per input, `group_by` separates candidates best.

A strong reasoning judge can cost more than the generation it judges: in one trial GLM-5.3 at
`reasoning_effort="low"` wrote about 7K output tokens per judge call (about 92K token units), more than writing the
problems did. Judge a few rows first and check the units column of the step report before judging the whole set.

The default rubric, `pyforger.filter.judge.DEFAULT_RUBRIC`, describes the five levels of the default `max_score=5` by
how serious the output's mistakes are (1: fails the task; 3: a mistake makes part of it wrong, misleading, or
incomplete; 4: nothing wrong or misleading, only minor shortcomings; 5: nothing a careful expert would change). The
default `score_threshold=4` keeps levels 4 and 5; 3 also keeps outputs a reviewer would have to fix. `rubric={1: "...",
..., N: "..."}` replaces it and must describe exactly the levels 1 to `max_score`; any other `max_score` needs one (and
a `score_threshold` when it is below 4).

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
    explanation_field="judge_feedback",
)
```

`group_by="question"` judges the rows that share a value together, e.g. `n=4` samples per prompt: one prompt shows the
input once and the candidates in `<candidate label="A">` tags, and the judge gives feedback and a score per candidate
on the same rubric (`{"candidates": [{"label": "A", "feedback": "...", "score": 4}, ...]}`). Seeing the alternatives
helps the judge notice what a candidate gets wrong; each candidate's feedback is asked to stand on its own, without
naming the others, since labels are shuffled. Each candidate is still kept by `score_threshold` and
`agreement_threshold`, so a group can keep all its rows or none; the result is the kept rows in input order, as
without `group_by`. Candidates are shuffled per group and judge `j` sees that order rotated by `j`, so each candidate
takes each position equally often. With the default prompt a group of one row gets the single-output prompt; with a
`prompt_template` it gets the group prompt with one candidate. A candidate the judge leaves out or scores unparseably
is no vote for that candidate only.

All judge calls (prompts × `num_judges`) go out in one `model.generate_batch` call, one `n=1` request each, so they
run concurrently; `num_judges > 1` needs `temperature > 0`. Judge answers are read leniently: only the text after a
closed `</think>` block (an unclosed `<think>` is unparseable), then the last JSON object with the expected key, also
inside code fences or prose (raw newlines in strings allowed), else the last `"score": <number>` pattern (per entry
for a group, in any key order). Scores keep
fractions (`4.9` stays `4.9`, so it fails `score_threshold=5`), and a string score gives its leading number (`"4/5"`
is 4). Unparseable answers count as no vote, and so does a request that still fails after the model's retries with a
transient error (a 504, a timeout): one warning in the step report names the count and the last error, and the result
is not cached, so a rerun asks again. Another warning counts the judge calls that gave no score by cause (failed,
unparseable, or stopped mid-thought in an unclosed `<think>`) and the rows they left unscored, e.g. with `group_by`
`2 of 10 judge calls gave no score (...) → 6 rows unscored`; calls cut off mid-thought are not counted again in the
op's warning about completions that stopped at `max_tokens`. A row error (a content-filter block or an over-long
prompt; see `generate_from_context` in [alter.md](alter.md)) counts as a failed request too, and the warning names its
reason, e.g. `1 prompt longer than the model's context window`; a row error on every prompt (at least 2) raises. Other
errors a retry cannot fix (a bad key, another 400, a spending cap or token-unit quota) still raise. The same holds for
every LLM filter: `by_judged_difficulty` leaves a row whose request failed unrated, `by_pass_at_k` and
`by_comparative_difficulty` grade a failed sample `None` (excluded), and `inspect.topic_based_clustering` does not ask
a failed request again, though it asks again after an unparseable answer: a failed shard summary is skipped, a failed
label request gives its document the fallback topic, and a failed codebook request keeps the last codebook, or raises
`RuntimeError` when there is none yet.

`prompt_template` replaces the prompt: a format string or a callable taking a dict. The placeholders are the row's
fields (for a group, its first row's) plus `{input}`, `{output}`, `{criteria}`, `{max_score}`, and `{rubric}` (the
rendered levels); with `group_by` every prompt, one-row groups included, also gets `{candidates}` (the tagged outputs,
also as `{output}`), `{labels}` (`"A, B, C"`), and `{num_candidates}`. Write literal braces as `{{ }}`. If the prompt
does not contain the JSON key in double quotes (`"score"`, or `"candidates"` for a group), the feedback-then-score
answer instruction is appended for you; it mentions the rubric only if the prompt shows `{rubric}`. With `group_by`, a
template whose own answer format has `"score"` but not `"candidates"` raises `ValueError`. The feedback is read from
`"feedback"`, or `"explanation"` for older templates.
`score_field="judge_score"` adds the judges' mean score to kept rows and `explanation_field="judge_feedback"` the list
of their feedback. Without these fields rows are unchanged.

## Difficulty filters

`filter.by_comparative_difficulty` and `filter.by_pass_at_k` ask models to solve the task in `input_field` and grade
each answer against `ground_truth_field` or with a verifier. `by_comparative_difficulty` is the medium-cost filter in
both cost and reliability: one prediction per row from `weak_model` and from `strong_model`, keeping rows only the
strong model gets right. `by_pass_at_k` is the most expensive but most reliable difficulty filter: `k` samples per
row in one `n=k` call, keeping rows whose pass rate (passed / valid grades) is in `[min_rate, max_rate]` (default
`[0.1, 0.9]`, inclusive).
`by_judged_difficulty` (least expensive, least reliable) only asks a judge how hard a task looks.

No answer format is imposed. The prompt is the task itself (`prompt_template` replaces it, e.g. to ask for a final
`ANSWER: ...` line). `input_field` may hold text or chat messages: a list of `{"role", "content"}` dicts, such as an
RL row's `prompt`, is sent as the conversation; a number is sent as its text; any other value raises `ValueError`.
`by_judged_difficulty`'s `task_field` takes the same values; the judge reads chat messages as text, each one written
as `<role>:` and its content on the next line, separated by blank lines.
Each completion is graded on what `extract_answer` takes from it. The default,
`pyforger.filter.model_difficulty.answer_segment`, is the text after the completion's closed `<think>` block, stripped
of surrounding whitespace. `metric` is `"exact_match"` (normalized equality with the reference), `"rouge_l"` (F1 at
least `rouge_l_threshold`), or a verifier `(prediction, row) -> bool | None`, e.g. `filter.program_verifier("tests")`,
which runs generated programs on the row's tests (see
[Running programs against tests](alter.md#running-programs-against-tests)).
`ground_truth_field` is needed only for `exact_match` and `rouge_l`; a verifier reads the row itself. Verifiers run
`verifier_workers` (default 8) calls at a time, so one must be safe to call from several threads. It is part of the
cache key, since a verifier with wall-clock timeouts (e.g. `program_verifier`) can grade differently under load.

By default the whole answer segment is compared, so for `exact_match` the task or `prompt_template` should ask for a
bare answer, or pass your own `extract_answer(completion) -> str | None`. It replaces the default and sees the raw
completion, thinking included; e.g. `lambda t: t.rsplit("ANSWER:", 1)[-1].strip()` keeps what follows the last
`ANSWER:`. An empty answer segment grades `None`, as does `extract_answer` or a verifier returning `None`: excluded,
not failed. A truncated completion (an opening `<think>` and no `</think>`: the model hit `max_tokens`) grades as
`truncated` says. `by_pass_at_k` defaults to `truncated="fail"`: such a sample grades `False` without calling
`extract_answer` or the verifier, since a task the model thinks itself to death on is hard, not ungradable. (Before,
it graded `None`; `truncated="exclude"` restores that.) `by_comparative_difficulty` defaults to `"exclude"`: a weak
model that ran out of tokens has not shown the row is hard. The step report counts truncated samples under the
filter's histogram, e.g. `5 of 12 samples truncated (<think> never closed), graded failed`; if that is most of them,
raise `max_tokens`. The op's warning about completions that stopped at `max_tokens` counts them too, since a bare op's
note is never shown, e.g. `78 of 264 completions ... hit max_tokens=16384 (72 mid-thought)`; so `strict=True` fails
on them. A sample whose request failed has no text, so it grades `None` (excluded),
never truncated; with a `Model` or `TinkerModel` (any model whose `generate_batch` takes `return_exceptions`) this
includes fan-out copies that failed, so a prompt that came back short is padded with `None`. Otherwise, a model that
returns a number of completions other than `n` for a prompt raises `ValueError`: a sample missing for any reason other
than a failed request is never padded.
`by_pass_at_k` needs all `k` grades valid unless `min_valid` allows fewer, and a row needs both comparative grades.
With `outcomes_field` set, kept rows carry their grades as `{group: [True | False | None, ...]}`, one grade per
sample: `{"model": [...]}` for `by_pass_at_k`, `{"weak": [...], "strong": [...]}` for `by_comparative_difficulty`.
With `completions_field` set, they carry the raw completions in the same shape and order (`None` for an empty one),
e.g. to keep a student's passing samples as STaR targets without sampling again. Predictions are cached apart from
grades, so changing the rate bounds or adding `completions_field` makes no model calls, and changing the metric
regrades the cached predictions. All prompts go out in one `generate_batch` call (`batch_size=None`, the default),
and each task is its own row in the request cache, so editing or adding tasks, or raising `k`, asks only for the new
samples.

```python
hard = filter.by_pass_at_k(
    rows,
    model=student,  # e.g. TinkerModel("Qwen/Qwen3-8B", max_tokens=16384); see acquire.md
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

To measure without filtering, keep every row with `min_rate=0.0, max_rate=1.0`. The step report's pass-rate
histogram, or each row's grades, is then the measurement; e.g. how often a model, such as a fine-tuned checkpoint,
already solves your prompts:

```python
measured = filter.by_pass_at_k(
    rows,
    model=checkpoint,
    input_field="prompt",  # text or chat messages
    metric=verifier,  # (prediction, row) -> bool | None; no ground_truth_field needed
    k=8,
    min_rate=0.0,
    max_rate=1.0,
    outcomes_field="grades",  # every row gets {"model": [True, False, ...]}
)
```

A row with too few valid grades (see `min_valid`) is still dropped.

`filter.by_judged_difficulty` asks an LLM judge to rate how hard each task in `task_field` is on a 5-point scale and
keeps rows rated in `[min_difficulty, max_difficulty]` (default `[3, 5]`, inclusive). It is the least expensive but
least reliable of the difficulty filters: one judge call per row, but the judge only guesses how hard the task is.
`by_comparative_difficulty` (medium cost and reliability) and `by_pass_at_k` (most expensive, most reliable) are the
next rungs up. The judge's fixed system prompt is anchored by three editable rubric descriptions, for levels 1, 3, and
5 (levels 2 and 4 sit between them); their defaults are module constants in `pyforger.filter.difficulty`
(`DEFAULT_MINIMUM_DIFFICULTY_CRITERION`, `DEFAULT_MEDIUM_DIFFICULTY_CRITERION`,
`DEFAULT_MAXIMUM_DIFFICULTY_CRITERION`). With `num_judges > 1` (which needs `temperature > 0`) a row's difficulty is
the mean of its judges' ratings. A row with no parseable rating is dropped. As with `by_llm_score`, ratings are cached
apart from the bounds, so changing `min_difficulty` or `max_difficulty` makes no new judge calls.

```python
hard = filter.by_judged_difficulty(
    rows,
    model=model,
    task_field="problem",
    min_difficulty=4,
    num_judges=3,
    temperature=0.7,
    maximum_difficulty_criterion="A problem that most olympiad medalists could not solve in an hour.",
)
```

`by_topic_frequency` balances topics. It clusters `field` into `num_clusters` topics with
[`inspect.topic_based_clustering`](inspect.md). A topic whose share of the input rows is above `max_frequency`
keeps a random subset of `max_frequency` of the input (seeded by `sample_seed`); a topic below `min_frequency` is
dropped (bounds inclusive). `1 / num_clusters` is the uniform share, and `max_frequency` defaults to twice that.
`topic_field="topic"` adds each kept row's topic name. `context` is an optional hint for topic names (see
`topic_based_clustering` in [inspect.md](inspect.md)). The clustering is the expensive part (one LLM call per unique
value) and is cached as its own op, so trying other bounds makes no new model or embedding calls; use the report's
histogram of topic shares to choose them.

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
