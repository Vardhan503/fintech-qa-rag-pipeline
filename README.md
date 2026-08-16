# Fin-Tech Q/A RAG Pipeline

Ingest the [FinQA](https://github.com/czyssrs/FinQA) development split into **page-level documents** and **structure-aware chunks** for a retrieve-then-calculate RAG pipeline over financial filings.

Raw FinQA stores one row per **question**. Many questions share the same 10-K/10-Q page. This repo collapses those repeats, cleans PDF extraction noise, and cuts each page the same way the dataset labels gold evidence: **one table row** or **one text line**.

```
883 questions  →  299 unique pages  →  8,532 chunks
```

## Why this chunking

LangChain recursive / character / token splitters cut by length. FinQA gold is already `ann_table_rows` / `ann_text_rows` (for example `table_3` = the American Express row). Length-based windows can split `637` from `5.0` or merge Visa and AmEx into one blob, which breaks retrieval eval.

Here a chunk is the same unit the benchmark calls gold:

| Question | Gold | Chunk id |
|---|---|---|
| average payment volume per transaction for American Express? | `ann_table_rows: [3]` | `V/2008/page_17.pdf::table_row::3` |

## Repository layout

```
data/
  raw/dev.json                 # FinQA dev split (883 QA examples)
  processed/
    documents.jsonl            # 299 unique PDF pages
    chunks.jsonl               # table-row + text-line chunks
data_preprocessing.ipynb       # EDA, cleaning, document + chunk build
```

### `documents.jsonl` — one row = one page

| Field | Meaning |
|---|---|
| `doc_id` | PDF page (`TICKER/YEAR/page_N.pdf`) |
| `pre_text` / `post_text` | Cleaned paragraphs above / below the table |
| `table` | Spreadsheet from `table_ori` with HTML stripped |
| `table_markdown` | Same table as readable text |
| `full_text` | `pre_text` + markdown table + `post_text` (one ingest blob) |
| `qa_ids` | Question ids on this page (join back to `dev.json` for answers) |

### `chunks.jsonl` — one row = one retrievable piece

| Field | Meaning |
|---|---|
| `chunk_id` | `{doc_id}::{table_row\|text_line}::{row_index}` |
| `doc_id` | Parent page |
| `chunk_type` | `table_row` or `text_line` |
| `row_index` | Table: 1-based (header is 0, never a chunk). Text: index in `pre_text + post_text` |
| `text` | Linearized sentence (table) or cleaned line (text) |

## What the notebook does

1. **Load** `data/raw/dev.json` and flatten QA metadata for EDA.
2. **Inspect noise** — lone `"."` lines, spaced `( 1 )` / `$ 2457`, duplicate pages.
3. **Clean** — drop punctuation-only lines, fix FinQA tokenization, strip `<i></i>` / `<sup>` from tables.
4. **Documents** — group by `filename`, write `data/processed/documents.jsonl`.
5. **Chunks** — linearize each table data row and each text line, write `data/processed/chunks.jsonl`.

Cleaning examples:

- `( billions )` → `(billions)`, `( 1 )` → `(1)`
- `$ 2457` → `$2457`
- Visa page 17 `pre_text`: 44 lines → 5 (39 lone `"."` lines removed)

Ragged tables (header shorter than a data row) keep extra cells as `col_N` instead of crashing.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook data_preprocessing.ipynb
```

Run the notebook top to bottom. It reads `data/raw/dev.json` and overwrites the processed jsonl files.

## Data

FinQA (Chen et al., EMNLP 2021): numerical reasoning over financial reports. This repo uses the **dev** split only. Source: [czyssrs/FinQA](https://github.com/czyssrs/FinQA).

## Next

Embed `chunks.jsonl`, retrieve by `chunk_id`, and score against `qa.ann_table_rows` / `ann_text_rows` from `dev.json`.
