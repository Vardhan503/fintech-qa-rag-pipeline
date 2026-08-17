# FinQA RAG — does structure-aware chunking actually help?

A controlled comparison of chunking strategies for retrieval-augmented question
answering over financial filings, run on the [FinQA](https://github.com/czyssrs/FinQA)
development split.

The interesting result is that the two evaluation layers disagree. Row-level
chunking retrieves the most precisely, but **whole-table chunking answers the most
questions correctly** — because most FinQA questions need two cells from the same
table, and only a whole-table chunk hands the model both at once. Retrieval
metrics alone would have picked the wrong strategy.

## Results

44-question stratified sample, `BAAI/bge-small-en-v1.5` for retrieval,
`qwen2.5:7b-instruct` (local Ollama) for generation, top-k = 5.

| Strategy | Chunks | P@5 | R@5 | R@10 | **Answer accuracy** |
|---|---|---|---|---|---|
| Row-level (baseline) | 8,931 | 0.123 | 0.474 | 0.557 | 40.9% |
| Naive fixed-size (control) | 2,689 | 0.237 | 0.765 | 0.815 | 29.5% |
| **Whole-table** | 7,575 | 0.205 | 0.709 | 0.798 | **45.5%** |
| Sentence-window | 8,931 | 0.328 | 0.652 | 0.722 | 36.4% |

![Recall@k by strategy](data/processed/chunking_comparison.png)

Reading the table: the naive control has the best *retrieval* recall yet the worst
*answer* accuracy. Long character windows overlap enough text to score well on
content-overlap matching while still handing the model a blob it can't compute
from. That gap is the whole argument for evaluating generation, not just retrieval.

### Table chunk size

Row-level and whole-table are the two ends of one dial, so
`scripts/run_chunk_size_sweep.py` measures the middle (retrieval only, recall@5
over the 818 questions that carry gold annotations):

| Rows per table chunk | Chunks | Recall@5 |
|---|---|---|
| 1 (row-level) | 8,931 | 62.8% |
| 2 | 8,189 | 67.7% |
| 3 | 7,918 | 70.4% |
| **5** | 7,718 | **71.3%** |
| whole table | 7,575 | 70.2% |

Recall climbs steeply from 1 to 3 rows and then flattens, so most of the benefit
is simply *not splitting related rows apart* rather than table-level context per
se. Five rows edges out the whole table by 1.1 points, which is inside the noise
on 818 questions — the honest read is that anything from 3 rows up performs about
the same, and one row per chunk is the clear loser.

## The dataset problem this project is really about

FinQA labels gold evidence as **positional indices** — `ann_table_rows: [3]` means
the fourth row of the table, `ann_text_rows: [7]` means the eighth line of
`pre_text + post_text`. Three consequences drove most of the design:

**1. Cleaning cannot drop lines.** The corpus has ~400 lines that are a lone `.`,
and deleting them shifts every later index, silently repointing gold labels at the
wrong sentence. `index_preserving_clean` is one-line-in, one-line-out and only
*flags* `is_noise`; flagged chunks are excluded at index time instead.
Locked in by `tests/test_cleaning.py`.

**2. Pages repeat.** 883 questions cover only 299 distinct pages, so documents are
grouped by filename with a `qa_ids` backlink rather than embedded once per question.

**3. `exe_ans`, not `answer`, is ground truth.** The human-typed `answer` field is
rounded and occasionally unrelated (`answer: '1%'` where `exe_ans: 0.015`). Grading
against it would mark correct responses wrong. `answers_match` also has to absorb
FinQA's percent-scale ambiguity — `exe_ans: 0.84882` versus a model saying `85%` —
and handle boolean questions where `exe_ans` is the string `"yes"`.
Locked in by `tests/test_answer_matching.py`.

## Layout

```
notebooks/
  01_preprocessing_eda.ipynb     # EDA + narrative; calls src/data/* only
  02_chunking_lab.ipynb          # strategy comparison; calls src/chunking/* + src/eval/*
src/
  data/       loader.py  cleaning.py  reconstruction.py
  chunking/   row_level.py  naive_fixed.py  whole_table.py  sentence_window.py
              row_group.py  parent_child.py
  eval/       gold_mapping.py  retrieval_harness.py  generation_harness.py  sampling.py
  utils/      io.py
scripts/      run_preprocessing.py  build_chunking_strategies.py
              run_retrieval_eval.py  run_generation_eval.py  build_comparison_report.py
tests/        pytest suite pinning the bugs already caught
data/
  raw/dev.json                   # read-only source
  processed/                     # documents, chunk stores, eval results, chart
```

Notebooks hold no logic — every function they call is imported from `src/`, so the
same code path runs interactively and in the scripts.

## Chunk schema

Every strategy emits the same keys, which is what lets one harness evaluate all of
them:

| Field | Meaning |
|---|---|
| `chunk_id` | `{doc_id}::{chunk_type}::{row_index}` |
| `chunk_type` | `table_row`, `text_line`, `whole_table`, `text_window`, `fixed_size`, `row_group` |
| `row_index` | Table rows count from 1 (row 0 is the header); text lines index into `pre_text + post_text` |
| `text` | Linearized sentence, raw line, or character window |
| `is_noise` | Flagged junk line — kept for index alignment, excluded from retrieval |
| `doc_id` | Parent page, `TICKER/YEAR/page_N.pdf` |

For the row-level strategy `row_index` is deliberately identical to FinQA's own
indexing, so a gold reference is just a formatted `chunk_id` and
`eval_dataset.jsonl` resolves with **0 unresolved references**.

## Two retrieval scorers, and when each is valid

- `evaluate_chunking_strategy` — exact `chunk_id` match. Only meaningful within the
  row-level id scheme; a `whole_table` chunk can never equal a `table_row` gold id.
- `evaluate_chunking_strategy_generic` — content-overlap match, valid *across*
  granularities. Use this for any cross-strategy comparison.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# generation eval only: a local Ollama serving the judge model
ollama pull qwen2.5:7b-instruct
```

## Reproduce

```bash
python scripts/run_preprocessing.py           # dev.json -> documents, chunks, eval_dataset
python scripts/build_chunking_strategies.py   # strategies 2-4 -> chunks_*.jsonl
python scripts/run_retrieval_eval.py          # -> eval_results_*.json          (~5 min)
python scripts/run_chunk_size_sweep.py        # -> eval_results_chunk_size.json (~2 min)
python scripts/run_generation_eval.py         # -> eval_results_generation.json (~15-20 min, needs Ollama)
python scripts/build_comparison_report.py     # table + chart
pytest                                        # 38 tests
```

The sample in `gen_eval_sample.jsonl` is reused if present, so every strategy —
including runs from earlier sessions — is graded on identical questions.

## Known gaps

- **Strategy 5 (parent-child) is not built.** `src/chunking/parent_child.py` holds
  the design: index row-level for retrieval precision, return the parent table as
  context for accuracy. It's the natural response to the row-level/whole-table
  split above.
- **`retrieval_hit_rate` in `run_generation_eval` uses exact id matching**, so it is
  not comparable across strategies with different id schemes. It does not affect
  `accuracy`.
- **`clean_line` only tightens parentheses around digits**, so `( billions )` stays
  spaced on ~2,100 lines. Changing it rewrites every chunk store, so it's pinned by
  a test rather than silently fixed.
- **65 of 883 questions carry no gold rows at all** in FinQA's annotations, which
  caps how high exact-id precision can read.
- Single embedding model, single judge LLM, one seed. The accuracy gaps between
  adjacent strategies are a few percentage points on 44 questions, so treat the
  ordering of the middle two as suggestive rather than settled.

## At 10x scale

The current pipeline holds every chunk vector in a NumPy array and re-embeds the
whole corpus on each run, which is fine for 8,931 chunks and wrong for 900,000:

- Persist embeddings and swap the brute-force dot product for an ANN index (FAISS
  HNSW, or pgvector if the metadata belongs in Postgres anyway).
- Add BM25 alongside the dense retriever. Financial questions lean on exact tokens
  — tickers, fiscal years, line-item names — where lexical matching beats
  embeddings, and hybrid fusion is the cheapest accuracy win available here.
- Move generation eval off sequential local calls to a batched, concurrent client;
  it currently dominates wall-clock time.
- Cache per-document chunk output keyed by a content hash so re-runs only touch
  changed filings.

## Data

FinQA (Chen et al., EMNLP 2021), dev split only. Source:
[czyssrs/FinQA](https://github.com/czyssrs/FinQA).
