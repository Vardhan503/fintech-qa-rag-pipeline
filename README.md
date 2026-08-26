# FinQA RAG Pipeline

End-to-end RAG over the [FinQA](https://github.com/czyssrs/FinQA) development
split. The production path is **whole-table chunks + Qdrant + Ollama**. The repo
also keeps the comparison experiments that led there: chunking strategies,
vector DBs, query transforms, and hybrid search.

## Final choices

| Decision | Choice | Why |
|---|---|---|
| **Chunking** | Whole-table | Best answer accuracy (45.5% on the 44-q sample) |
| **Vector DB** | Qdrant | Same recall as FAISS/Chroma; persists on disk |
| **Embeddings** | `BAAI/bge-small-en-v1.5` | Shared across eval and RAG |
| **LLM** | `qwen2.5:7b-instruct` (Ollama) | Local generation |
| **Query transform** | Baseline dense (no HyDE) | HyDE hurt recall (~0.28 vs ~0.50); decompose was only a small lift |
| **Hybrid / rerank** | Dense-only for now | Dense NDCG@10 beat BM25 and 50/50 hybrid on this sample |

---

## Pipeline

```
data/raw/dev.json
  → preprocess          documents + gold chunks + eval set
  → build strategies    naive / whole-table / sentence-window / row-group-5
  → retrieval + generation eval
  → vector DB bakeoff   FAISS / Chroma / Qdrant
  → query transform     HyDE / multi-query / step-back / decompose
  → hybrid search       BM25 / dense / EnsembleRetriever / rerank
  → RAG                 whole-table → qdrant_db_final → Ollama
```

---

## Setup

```bash
conda create -n finqa-rag python=3.12
conda activate finqa-rag
conda install -c conda-forge faiss-cpu numpy libgfortran   # Apple Silicon
pip install -r requirements.txt

ollama pull qwen2.5:7b-instruct
# ollama is usually already running as a service; only run `ollama serve` if not
```

Put FinQA `dev.json` at `data/raw/dev.json` if missing.

---

## Reproduce

```bash
# Core corpus
python scripts/run_preprocessing.py
python scripts/build_chunking_strategies.py

# Chunking comparison
python scripts/run_retrieval_eval.py
python scripts/run_chunk_size_sweep.py          # optional
python scripts/run_generation_eval.py           # needs Ollama
python scripts/build_comparison_report.py

# Vector DB bakeoff (whole-table chunks)
python scripts/vector_db.py

# Query transform + hybrid (row-level index, in-memory Qdrant)
python scripts/run_query_transform_eval.py      # needs Ollama; slow
python scripts/run_hybrid_search_eval.py        # downloads cross-encoder once

# Production RAG
python scripts/run_rag.py --setup
python scripts/run_rag.py --ask "what is the average payment volume per transaction for american express?"

pytest
```

---

## Results (summary)

### Chunking (content-overlap retrieval + 44-q generation)

| Strategy | R@5 | Answer accuracy |
|---|---|---|
| Row-level | 0.628 | 40.9% |
| Naive fixed-size | **0.765** | 29.5% |
| **Whole-table** | 0.709 | **45.5%** |
| Sentence-window | 0.654 | 36.4% |
| Row-group-5 | 0.713 | (optional re-run) |

### Vector DBs (whole-table, same embeddings)

| Store | Recall@5 | Notes |
|---|---|---|
| FAISS | ~65.7% | Fast; in-memory |
| ChromaDB | ~65.6% | On disk |
| **Qdrant** | ~65.7% | On disk — production |

### Query transform (64-q stratified sample, exact-id R@5)

| Method | R@5 (ALL) |
|---|---|
| Baseline dense | **0.50** |
| HyDE | 0.28 |
| Multi-query | 0.48 |
| Step-back | 0.49 |
| Decompose | 0.51 |

### Hybrid search (same sample, NDCG@10)

| Method | NDCG@10 (ALL) |
|---|---|
| **Dense only** | **0.45** |
| BM25 only | 0.16 |
| Hybrid 50/50 | 0.33 |
| Hybrid 70/30 | 0.19 |
| Hybrid + rerank | 0.24 |

Numbers live in `data/processed/eval_results_*.json`. Re-run the scripts to refresh.

---

## Layout

```
src/
  data/             load, clean, reconstruct pages
  chunking/         row_level, naive_fixed, whole_table, sentence_window, row_group
  eval/             gold map, retrieval/generation harness, experiment helpers
  query_transform/  hyde, multi_query, step_back, decompose, rewrite
  hybrid_search/    bm25+dense hybrid, reranker, ndcg
  retrieval/        shared RRF
  rag/              production Qdrant + Ollama pipeline
  models.py         embedder + LLM factories
  utils/            path-aware jsonl I/O

scripts/            preprocess, build strategies, evals, vector_db, rag
tests/
data/raw/dev.json
data/processed/     chunks, eval JSON, qdrant_db_final/
```

---

## Chunk schema

| Field | Meaning |
|---|---|
| `chunk_id` | `{doc_id}::{chunk_type}::{row_index}` |
| `chunk_type` | `table_row`, `text_line`, `whole_table`, `text_window`, `fixed_size`, `row_group` |
| `is_noise` | Kept for index alignment; dropped from the retrievable index |
| `doc_id` | Parent page path |

Cross-strategy retrieval uses **content overlap** against row-level gold text.
Query-transform / hybrid scripts that share the row-level id scheme can use
exact id match.

---

## Notes

- Cleaning is index-preserving: FinQA gold rows are positional indices.
- Grade answers against `exe_ans`, not the noisy `answer` string.
- 65/883 questions have no gold evidence and are skipped in scoring.
- FAISS in the bakeoff is in-memory only; production persistence is Qdrant.

## Data

FinQA (Chen et al., EMNLP 2021), dev split.
[czyssrs/FinQA](https://github.com/czyssrs/FinQA)
