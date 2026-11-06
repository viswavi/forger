# How the rl_math dataset was decided

`pipeline.py` is where this dataset ended up. This file is how it got there: at each decision, the options we
weighed, the Forger measurement that told them apart, what we chose, and what result would change our mind.

> **The numbers here are illustrative.** They were written for this example, not recorded from a run. Their shape is
> what these reports print, and each measurement can be rerun with the call shown. In your own log, record the numbers
> you measured, and say which ones you guessed.

**Goal.** An RL dataset for Qwen3-8B with a reward a program can check: competition-style math, one integer answer per
problem, reward 1 for the right final answer and 0 otherwise. RL samples 8 responses per prompt (GRPO), so a prompt
whose 8 samples all score the same teaches nothing. Every decision answers one of four questions:

1. **Hard enough?** Does the student solve some samples of each problem, but not all of them?
2. **Representative?** Do the problems look like the math we want the student to get better at?
3. **Diverse?** Are there many distinct problems, rather than a few templates?
4. **Sound reward?** Does the reward pay for solving the problem and nothing else?

| Step         | In    | Out   | Decided in |
|--------------|-------|-------|------------|
| sources      | 560   | 431   | D1, D5     |
| write        | 431   | 1,188 | D1, D5     |
| verify       | 1,188 | 974   | D2         |
| dedupe       | 974   | 892   | D3         |
| calibrate    | 892   | 508   | D4         |
| check_reward | 508   | 508   | D5         |
| finish       | 508   | 508   | D6         |

## D1. Where the problems come from (representative, diverse)

The reference for "representative" is 560 problems from the public training split that we never use as seeds; the test
split is held out for decontamination only (D3). We built a 200-row pilot from each candidate source and measured each
one the same way:

```python
inspect.embedding_dispersion(pilot, "problem", reference=reference, duplicate_threshold=0.9)
filter.by_pass_at_k(pilot, model=student, input_field="prompt", k=8, min_rate=0, max_rate=1, histogram=True, ...)
```

| Source                        | Coverage | Off-dist. | Near ref. | Distinct of 200   | Vendi (ref. 58) | Solved 8/8 |
|-------------------------------|----------|-----------|-----------|-------------------|-----------------|------------|
| A. Training problems as is    | 0.96     | 0.03      | 0.11      | 198               | 57              | 46%        |
| B. LLM-written from topics    | 0.52     | 0.09      | 0.04      | 181               | 31              | 24%        |
| C. Templated generators       | 0.31     | 0.22      | 0.02      | ! 14 (largest 22) | 6               | 12%        |
| D. LLM rewrites of seeds      | 0.88     | 0.05      | 0.16      | 191               | 49              | 22%        |

- **A** looks like the reference because it is the reference's own distribution. But the student already solves 46% of
  these problems on all 8 samples. It has most likely trained on this public set, so those rows give no gradient.
- **B** collapsed onto a few topics. Its `not covered:` examples were almost all geometry and precalculus, and its
  Vendi score is half the reference's.
- **C** is the trap a diversity score alone hides. Every problem is textually different, but the near-duplicate groups
  say "200 rows hold about 14 distinct ones": each template is one problem with its numbers changed.
- **D** keeps most of A's coverage, and the student has no memorized answers for it.

**Chose D**, with three rewrites per seed. Its near-reference share (0.16, against about 0.10 for rows drawn like the
reference) means some rewrites stay close to their seed. We handled that with a stronger "change the objects, the
numbers and what is asked" instruction and with D3's threshold, not by changing the source.
*Would change it:* a near-reference share above 0.25 after D3, meaning the rewrites are paraphrases.

## D2. Is the stated answer right? (sound reward)

The reward compares against the writer's answer, so a wrong answer trains the student to be wrong. The options:

- trust the writer;
- have the writer solve its own problem 4 times and require agreement;
- have an independent model solve it 4 times and require agreement;
- check the answer with a symbolic solver (works for only some problem types).

We drew 300 drafts and ran an independent verifier, then read every row where the verifier's majority disagreed with
the writer (51 rows, 17%): the writer was wrong on 36, the verifier on 11, and 4 problems were ambiguous. On 20 of the
36 rows where the writer was wrong, the writer also agreed with itself, repeating the same mistake. So self-agreement
was ruled out, and **writer and verifier are different models**. The agreement threshold came from the histogram:

```python
filter.by_pass_at_k(drafts, model=verifier, input_field="prompt", k=4, min_rate=0.75, max_rate=1.0,
                    metric=correct, extract_answer=final_answer, histogram=True)
```

```text
filter.by_pass_at_k · pass rate · keeps 0.75 to 1: 974 of 1,188 (0.82)
    pass rate  rows                                kept
            0   107  ███▌                          no
         0.25    48  █▌                            no
          0.5    59  █▉                            no
         0.75   107  ███▌                          yes
            1   867  ████████████████████████████  yes
```

Of 20 rows we read at 2/4, 11 had a wrong answer, so 2/4 is dropped. At 3/4, 2 of 20 did, so 3/4 is kept.
**Chose ≥ 3/4** (yield 0.82). *Would change it:* an audit that finds more than 1 in 10 wrong at 3/4.

## D3. How close counts as a duplicate (diverse)

Forger's default threshold of 0.95 is calibrated on long coding problems, but these are one-paragraph problems. We
measured the verified rows at three thresholds:

```python
for t in (0.95, 0.9, 0.85):
    inspect.embedding_dispersion(verified, "problem", duplicate_threshold=t)
```

| Threshold | Share with a near-copy | Groups | Largest | What the listed pairs were                         |
|-----------|------------------------|--------|---------|----------------------------------------------------|
| 0.95      | 0.04                   | 19     | 2       | Exact restatements                                 |
| 0.90      | 0.13                   | 52     | 4       | Siblings from one seed with the numbers swapped    |
| 0.85      | ! 0.31                 | 140    | 19      | Distinct problems opening "How many positive…"     |

At 0.9 the groups are real copies: siblings that differ only in numbers. At 0.85, the largest group joins problems
that merely open alike. **Chose 0.9**, applied to the rows and, against the test split, for decontamination. That
removed 75 copies and 7 rows close to a test problem.
*Would change it:* a largest group above 5 at 0.9 in a later run, which would mean the rewrite prompt is producing
templates.

## D4. Which difficulties to keep (hard enough)

We measured the student at RL's own settings: 8 samples at temperature 1.0, with a 16,384-token limit. 29% of rows
are solved 8/8, up from the pilot's 22%: verification removed problems with wrong answers, which no sample solves.

```python
filter.by_pass_at_k(deduped, model=student, input_field="prompt", k=8, min_rate=0, max_rate=1,
                    metric=correct, extract_answer=final_answer, histogram=True)
```

```text
filter.by_pass_at_k · pass rate · keeps 0 to 1: 892 of 892 (1.00)
    pass rate  rows                                kept
            0   125  █████████████▌                yes
         0.12    58  ██████▎                       yes
         ...
         0.88    89  █████████▋                    yes
            1   259  ████████████████████████████  yes
  431 of 7136 samples truncated (<think> never closed), graded failed; raise max_tokens if that is too many
  384 of 892 rows gave every sample the same grade (125 all 0, 259 all 1): no GRPO gradient there
```

Without a filter, 43% of RL's groups would carry no gradient. Since k equals RL's group size, the 0 and 1 bars predict
those dead groups directly. The options were:

- no filter;
- [1/8, 7/8];
- [1/8, 3/4], leaning harder;
- also keeping the 0/8 rows.

The 0/8 rows are hard, not broken: the verifier solved 81% of them at 4/4. But at k=8 they give no gradient until the
policy improves, so they are saved as a separate stretch set rather than mixed in. Leaning harder would cut the yield
from 508 to 419 rows, and nothing we had measured favored it.
**Chose [1/8, 7/8]** (508 rows). The 6% truncation is below where we would raise `max_tokens`.
*Would change it:* RL's training reward passing 0.8 early. That is saturation, and the answer is to re-measure with
the checkpoint and calibrate again.

## D5. Does the reward pay only for solving? (sound reward)

The first grader took any `\boxed{}` in the response that matched the answer. Before RL, we graded responses it should
pay for and responses it should not:

```python
inspect.rubric_scores(rows, reward_v1, n=200, response_fields=["reference_solution", "baseline_solution"],
                      probes=probes(), model=student, k=8)
```

| Column                  | v1: any box | v2: last box after `</think>` |
|-------------------------|-------------|-------------------------------|
| reference_solution      | 1.00        | 1.00                          |
| baseline_solution       | 0.58        | 0.56                          |
| sampled ×8              | 0.61        | 0.55                          |
| probe every_box         | ! 0.71      | 0.00                          |
| probe unclosed_thinking | ! 1.00      | 0.00                          |
| probe number_from_statement | 0.06    | 0.00 (rows dropped, below)    |
| empty / random / other_row | 0.00     | 0.00                          |

- **every_box** boxes every integer from -10 to 100. v1 paid for it on 71% of rows, because most answers are small. A
  policy would find that quickly.
- **unclosed_thinking** is a truncated response that names the answer inside its thinking. v1 paid for it too; v2
  reads only after `</think>`, the same rule our evaluation uses.
- **number_from_statement** scored 0.06 under v1, so 6% of problems printed their own answer in the statement. Rather
  than make the reward stricter, the write step now drops those rows (`answer not in statement`).

Under v2, the paired comparison read:

```text
  baseline_solution: below reference_solution on 88, tied 112, above 0 of 200 rows, mean gap +0.44
  sampled ×8: 9/200 prompts gave all samples the same score (5 all 0, 4 all 1): no RL gradient there
```

With a 0/1 reward, a tie means both responses were right (the baseline, which doesn't think, solves 56%). So the
reward separates the two on every row where they differ.
**Chose v2.** We rejected an LLM judge of answer equivalence (1/2 against 0.5): with integer-only answers, exact match
is already exact. The price is coverage: requiring an integer answer drops 23% of seeds (D6).
*Would change it:* a probe above 0.05 under v2, or an evaluation whose answers are not integers.

## D6. What the filters cost (representative)

```python
inspect.embedding_dispersion(final, "problem", reference=reference, duplicate_threshold=0.9)
```

Coverage fell from the pilot's 0.88 to 0.79. Off-distribution is 0.04, the near-duplicate share 0.03 (9 groups,
largest 2), and Vendi 44 against the reference's 58. The `not covered:` examples are precalculus identities and
geometry with radicals. Two decisions caused that: the integer-only answers (D5) and dropping problems solved 8/8 (D4).
**Accepted**, with the gap written down.
*Would change it:* an evaluation that weights precalculus. That would need non-integer answers and the equivalence
judge we rejected in D5.

## What we did not measure

- **Whether RL on these rows improves held-out scores.** Every check above is a proxy for that, measured before
  training.
- **How often the verifier and the writer agree on a wrong answer.** The D2 audit estimates it only from the rows we
  read.
- **Solution diversity.** Two problems can differ in wording and still need the same three steps. Embeddings of the
  problem text do not show that.
