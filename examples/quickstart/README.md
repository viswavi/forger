# Quick start: a trivia dataset in two versions

Two small pipelines that use Claude through LiteLLM. Each run costs about $0.16 and takes under a minute.

## Setup

```bash
git clone -b release https://github.com/viswavi/forger.git && cd forger
pip install -e .
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env         # in the repository root; .env is gitignored
cd examples/quickstart
```

Haiku 5.5 writes the questions, and Sonnet 5.5 judges or checks them.

## Version 1: `trivia_v1.py`

Asks one prompt 50 times ("Write one hard trivia question about astronomy, and its short answer"). Then it drops
near-duplicate questions (ROUGE-L) and keeps those an LLM judge scores 4 or 5 of 5.

```bash
FORGER_LIMIT=3 python trivia_v1.py   # every op on 3 rows: catches mistakes in seconds, for about a cent
python trivia_v1.py                  # the full run
```

Read the step reports. In our run, they showed two problems:

- `repeated: question "white dwarf" 12/35 (34%)`. The same few facts come back again and again: about 10 of the 35
  questions asked for the Chandrasekhar limit, and 7 for Bessel's parallax of 61 Cygni. ROUGE-L removed 15 of 50,
  but it cannot see the same fact asked in other words.
- The judge's histogram: all 35 questions scored 5. The quality filter removed nothing.

## Version 2: `trivia_v2.py`

Three steps, each a fix for what version 1's report showed:

1. `subtopics` asks once for 40 distinct subtopics of astronomy.
2. `write_questions` writes one question per subtopic, then drops questions that repeat an answer.
3. `check_answers` has a second model answer each question without seeing the answer, and keeps the question only if
   a judge says the two answers name the same thing.

```bash
FORGER_LIMIT=3 python trivia_v2.py
python trivia_v2.py
```

In our run, all 40 questions had different answers. The checker agreed with all 40, and the step report warned
`yield 1.00 above max_yield 0.98`: the check did not remove anything. Is the checker too easy to convince, or
are the questions too easy for Sonnet? That is the next question to look into. For example, try a weaker checker
(Haiku), or ask for harder questions.

## Ideas to try

- Change `TOPIC` in `trivia_v2.py`.
- Put `@step(sample=5)` on a step to see five sample rows in its report.
- Run `python -c "from pyforger import inspect; inspect.inspect_step('check_answers')"` to print a step's report
  again, with more sample rows.
- Rerun after an edit: only the steps you changed call the API again. The rest come from the cache.
