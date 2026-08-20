# FinQA RAG Pipeline

End-to-end RAG over the [FinQA](https://github.com/czyssrs/FinQA) development
split: ingest financial filings, **compare chunking strategies**, **compare
vector databases**, then answer questions with a local LLM.

## Final choices (production)


| Decision      | Choice                                                     | Why                                                                                    |
| ------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **Chunking**  | **Whole-table**                                            | Best answer accuracy (45.5%) — most FinQA questions need two cells from the same table |
| **Vector DB** | **Qdrant**                                                 | Same recall as FAISS/Chroma (~65.7%), and persists to disk for the RAG app             |
| **RAG**       | Whole-table chunks + Qdrant + Ollama `qwen2.5:7b-instruct` | Embeddings: `BAAI/bge-small-en-v1.5`                                                   |


Retrieval metrics and answer accuracy disagree: naive fixed-size wins *retrieval*
recall but loses on *answers*. Generation accuracy is what drove the chunking
choice; on-disk persistence drove the vector DB choice (FAISS is in-memory only).

---



## Pipeline overview

```
data/raw/dev.json
   │
   ▼  1. Preprocess
documents.jsonl  chunks.jsonl (row-level gold)  eval_dataset.jsonl
   │
   ▼  2. Build alternate chunk stores
chunks_naive_fixed / whole_table / sentence_window / grouped_5
   │
   ▼  3. Evaluate chunking (retrieval + generation)
eval_results_*.json
   │
   ▼  4. Vector DB bakeoff (same whole-table chunks)
FAISS vs ChromaDB vs Qdrant
   │
   ▼  5. Production RAG
Qdrant (qdrant_db_final/) + Ollama
question → retrieve top-k → generate → answer
```

---



## Setup

```bash
# Python environment (conda recommended on Apple Silicon)
conda create -n finqa-rag python=3.12
conda activate finqa-rag

# FAISS: use conda on arm64 (pip wheel can segfault)
conda install -c conda-forge faiss-cpu numpy libgfortran

pip install -r requirements.txt

# Generation eval + RAG need a local Ollama model
ollama pull qwen2.5:7b-instruct
ollama serve   # if not already running
```

On Apple Silicon, embeddings run on CPU (`device="cpu"`) — MPS can crash mid-encode.
The scripts already do this.

Place FinQA `dev.json` at `data/raw/dev.json` if it is not already there.

---



## Reproduce (full evaluation)

Run from the project root:

```bash
# 1. Preprocessing (documents + row-level gold chunks + eval set)
python scripts/run_preprocessing.py

# 2. Alternative chunk stores (naive, whole-table, sentence-window, row-group-5)
python scripts/build_chunking_strategies.py

# 3. Retrieval eval across all 5 strategies (~5 min)
python scripts/run_retrieval_eval.py

# 4. Optional: table row-group size sweep (~2 min, no LLM)
python scripts/run_chunk_size_sweep.py

# 5. Generation eval (~15-20 min, needs Ollama)
python scripts/run_generation_eval.py

# 6. Summary table + recall@k chart
python scripts/build_comparison_report.py

# 7. Vector DB benchmark on whole-table chunks (~2 min)
python scripts/vector_db.py

# 8. Production RAG — build index once, then ask
python scripts/run_rag.py --setup
python scripts/run_rag.py --ask "what is the average payment volume per transaction for american express?"

# Tests
pytest
```

The sample in `gen_eval_sample.jsonl` is reused if present, so every strategy is
graded on identical questions across runs.

---



## Results



### 1. Chunking strategies

`BAAI/bge-small-en-v1.5` for retrieval (content-overlap scoring),
`qwen2.5:7b-instruct` for generation on a fixed **44-question** stratified sample,
top-k = 5.


| Strategy                   | Chunks | P@5       | R@5       | R@10      | **Answer accuracy** |
| -------------------------- | ------ | --------- | --------- | --------- | ------------------- |
| Row-level (baseline)       | 8,931  | 0.238     | 0.628     | 0.702     | 40.9%               |
| Naive fixed-size (control) | 2,689  | 0.237     | **0.765** | **0.815** | 29.5%               |
| **Whole-table ← chosen**   | 7,575  | 0.205     | 0.709     | 0.798     | **45.5%**           |
| Sentence-window            | 8,931  | **0.329** | 0.654     | 0.722     | 36.4%               |
| Row-group-5                | 7,718  | 0.231     | 0.713     | 0.790     | 45.7                |


**Takeaway:** highest retrieval recall ≠ best answers. Whole-table hands the LLM
both cells it needs in one chunk.

#### Table chunk-size sweep (retrieval only, recall@5)


| Rows per table chunk | Chunks | Recall@5  |
| -------------------- | ------ | --------- |
| 1 (row-level)        | 8,931  | 62.8%     |
| 2                    | 8,189  | 67.7%     |
| 3                    | 7,918  | 70.4%     |
| **5**                | 7,718  | **71.3%** |
| whole table          | 7,575  | 70.2%     |


Five rows edges whole-table by ~1 pp on retrieval, but whole-table still wins
generation — so production stays on whole-table.

---



### 2. Vector database comparison

Same **whole-table** chunks (7,176 indexable), same embeddings, 883 questions,
content-overlap recall@5:


| Tool                | Build time | Search time (883 q) | Recall@5 |
| ------------------- | ---------- | ------------------- | -------- |
| FAISS               | ~0.00s     | ~0.01s              | 65.7%    |
| ChromaDB            | ~2.7s      | ~0.2s               | 65.6%    |
| **Qdrant ← chosen** | ~4.5s      | ~5.2s               | 65.7%    |


All three tie on recall (same vectors, same math). FAISS is fastest but
**in-memory only**. Chroma and Qdrant persist under `data/processed/`. Production
RAG uses **Qdrant** at `data/processed/qdrant_db_final/`.

Re-run with `python scripts/vector_db.py` (writes `eval_results_vector_db.json`).

---



### 3. RAG (final system)

```
question
  → embed (BGE small)
  → Qdrant top-5 over whole-table chunks
  → ChatOllama (qwen2.5:7b-instruct)
  → ANSWER: line
```

```bash
python scripts/run_rag.py --setup --ask "your question"
python scripts/run_rag.py --ask "another question"   # index already on disk
```

```python
from src.rag.pipeline import SimpleRAG

rag = SimpleRAG()
rag.setup()                              # whole-table → qdrant_db_final/
result = rag.ask("your question here")
print(result["answer"])
print(result["sources"])
rag.close()
```

---



## Chunking strategies (what each does)


| #   | Strategy         | Module               | Idea                                                                          |
| --- | ---------------- | -------------------- | ----------------------------------------------------------------------------- |
| 1   | Row-level        | `row_level.py`       | One chunk per table row / text line — gold-aligned baseline                   |
| 2   | Naive fixed-size | `naive_fixed.py`     | LangChain `RecursiveCharacterTextSplitter` (500/50) — structure-blind control |
| 3   | **Whole-table**  | `whole_table.py`     | Entire table as one chunk — **production**                                    |
| 4   | Sentence-window  | `sentence_window.py` | Text line ± 1 neighbour; table rows unchanged                                 |
| 5   | Row-group-5      | `row_group.py`       | Group 5 table rows per chunk                                                  |


Parent-child was evaluated earlier and dropped from this repo to keep the
pipeline simpler; whole-table already delivers full-table context to the LLM.

---



## Project layout

```
src/
  data/          loader.py  cleaning.py  reconstruction.py
  chunking/      row_level.py  naive_fixed.py  whole_table.py
                 sentence_window.py  row_group.py
  eval/          gold_mapping.py  retrieval_harness.py
                 generation_harness.py  sampling.py
  rag/           pipeline.py          # Qdrant + ChatOllama
  models.py      HuggingFaceEmbeddings + ChatOllama
  utils/         io.py

scripts/
  run_preprocessing.py             # stage 1
  build_chunking_strategies.py     # stage 2
  run_retrieval_eval.py            # stage 3a
  run_chunk_size_sweep.py          # stage 3b (optional)
  run_generation_eval.py           # stage 3c
  build_comparison_report.py       # stage 3d
  vector_db.py                     # stage 4 — FAISS / Chroma / Qdrant
  run_rag.py                       # stage 5

tests/
data/
  raw/dev.json
  processed/                       # chunks, eval JSON, vector DBs
```



## Custom vs LangChain

**LangChain:** naive splitter, embeddings, ChatOllama, RAG LCEL chain, FAISS /
Chroma / Qdrant wrappers.

**Custom (FinQA-specific):** index-preserving cleaning, ragged table
linearization, gold `chunk_id` mapping, `answers_match` vs `exe_ans`,
table-aware chunkers.

---



## Chunk schema


| Field        | Meaning                                                                             |
| ------------ | ----------------------------------------------------------------------------------- |
| `chunk_id`   | `{doc_id}::{chunk_type}::{row_index}`                                               |
| `chunk_type` | `table_row`, `text_line`, `whole_table`, `text_window`, `fixed_size`, `row_group`   |
| `row_index`  | Table rows count from 1 (header is 0); text lines index into `pre_text + post_text` |
| `text`       | Linearized sentence, raw line, character window, or pipe-delimited table            |
| `is_noise`   | Flagged junk — kept for index alignment, excluded from retrieval                    |
| `doc_id`     | Parent page, e.g. `V/2008/page_17.pdf`                                              |


Cross-strategy retrieval uses **content overlap**, not exact id match — a
`whole_table` chunk can never equal a `table_row` gold id.

---



## Dataset constraints that drove the design

1. **Cleaning cannot drop lines.** FinQA gold labels are positional indices.
  `index_preserving_clean` is one-line-in, one-line-out and only flags `is_noise`.
2. **Pages repeat.** 883 questions → 299 unique pages; documents are grouped by
  filename with a `qa_ids` backlink.
3. **Grade against** `exe_ans`**, not** `answer`**.** The typed `answer` field is noisy
  (`answer: '1%'` where `exe_ans: 0.015`).

---



## Known gaps

- Accuracy gaps between adjacent strategies are a few points on 44 questions —
treat mid-pack ordering as suggestive.
- 65 of 883 questions carry no gold rows in FinQA's annotations.
- FAISS in the bakeoff is in-memory; production persistence is Qdrant.
- Single embedding model, single judge LLM, one sample seed.

---



## Data

FinQA (Chen et al., EMNLP 2021), dev split only.
Source: [czyssrs/FinQA](https://github.com/czyssrs/FinQA).