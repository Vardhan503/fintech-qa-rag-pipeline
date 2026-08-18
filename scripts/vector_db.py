# Simple test: load the same chunks into 3 different search tools,
# and compare how fast + accurate each one is.

import os
import sys
import time
from pathlib import Path

# single-threaded BLAS/OpenMP — prevents threading races on macOS arm64
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# repo root is one level above scripts/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

import faiss  # import AFTER sentence_transformers — avoids arm64 BLAS conflict
import chromadb
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.utils.io import load_jsonl

# force CPU — MPS (Apple Silicon GPU) crashes mid-encode
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")

# ---------- Step 1: load chunks ----------
chunks = [c for c in load_jsonl("data/processed/chunks_whole_table.jsonl")
          if not c.get("is_noise", False)]
eval_examples = load_jsonl("data/processed/eval_dataset.jsonl")

# NEW: this is the fix. Our gold_chunk_ids are written in ROW-LEVEL format
# (doc::table_row::3), but whole-table chunks have DIFFERENT ids
# (doc::whole_table::0) — they can never match by ID even when correct.
# So instead of comparing IDs, we look up what TEXT each gold_chunk_id
# actually points to, and check if that text shows up inside whatever
# we retrieved. This works no matter what chunking strategy we're testing.
row_level_chunks = load_jsonl("data/processed/chunks.jsonl")
gold_text_lookup = {c["chunk_id"]: c["text"] for c in row_level_chunks}

chunk_ids = [c["chunk_id"] for c in chunks]
chunk_texts = [c["text"] for c in chunks]
question_texts = [ex["question"] for ex in eval_examples]

print(f"Loaded {len(chunks)} chunks and {len(eval_examples)} questions.")

# ---------- Step 2: embed once, reuse for all three tools ----------
print("Embedding chunks...")
chunk_vectors = np.asarray(
    model.encode(chunk_texts, normalize_embeddings=True,
                 batch_size=64, show_progress_bar=False,
                 convert_to_tensor=False),
    dtype=np.float32,
).copy()

print("Embedding questions...")
question_vectors = np.asarray(
    model.encode(question_texts, normalize_embeddings=True,
                 batch_size=64, show_progress_bar=False,
                 convert_to_tensor=False),
    dtype=np.float32,
).copy()

dimension = chunk_vectors.shape[1]
print(f"chunk_vectors : {chunk_vectors.shape}  dtype={chunk_vectors.dtype}  "
      f"C-contiguous={chunk_vectors.flags['C_CONTIGUOUS']}")
print(f"question_vecs : {question_vectors.shape}")

# ---------- Step 3: content-based Recall@5 (the fix) ----------
def word_overlap(retrieved_text, gold_text, threshold=0.6):
    """Simple check: does the retrieved text contain most of the gold
    answer's words? Works across DIFFERENT chunk id schemes."""
    gold_words = set(w.lower().strip(".,;:()") for w in gold_text.split())
    retrieved_words = set(w.lower().strip(".,;:()") for w in retrieved_text.split())
    if not gold_words:
        return False
    overlap = len(gold_words & retrieved_words) / len(gold_words)
    return overlap >= threshold


def check_recall(get_top5_texts_fn):
    """get_top5_texts_fn(question_index) -> list of 5 text strings (not ids!)"""
    found = 0
    scorable = 0
    for i, ex in enumerate(eval_examples):
        gold_ids = ex["gold_chunk_ids"]
        gold_texts = [gold_text_lookup[g] for g in gold_ids if g in gold_text_lookup]
        if not gold_texts:
            continue
        scorable += 1
        top5_texts = get_top5_texts_fn(i)
        found_here = False
        for retrieved_text in top5_texts:
            for gold_text in gold_texts:
                if word_overlap(retrieved_text, gold_text):
                    found_here = True
        if found_here:
            found += 1
    return found / scorable if scorable else 0.0


results = {}

# ==================================================================
# TOOL 1: FAISS — in-memory, fastest build + search
# ==================================================================
print("\n--- Testing FAISS ---")

t0 = time.time()
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(chunk_vectors)
faiss_build_time = time.time() - t0
print(f"  index built ({faiss_index.ntotal} vectors)")

t0 = time.time()
_, faiss_indices = faiss_index.search(question_vectors, 5)
faiss_search_time = time.time() - t0

def faiss_get_top5_texts(qi):
    return [chunk_texts[i] for i in faiss_indices[qi]]

faiss_recall = check_recall(faiss_get_top5_texts)
results["FAISS"] = {"build_time": faiss_build_time, "search_time": faiss_search_time, "recall": faiss_recall}
print(f"  build={faiss_build_time:.2f}s  search={faiss_search_time:.2f}s  recall={faiss_recall:.1%}")


# ==================================================================
# TOOL 2: ChromaDB — saves to disk, simple Python API
# ==================================================================
print("\n--- Testing ChromaDB ---")

chroma_client = chromadb.PersistentClient(
    path=str(ROOT / "data" / "processed" / "chroma_db"))
try:
    chroma_client.delete_collection("finqa_chunks")
except Exception:
    pass
chroma_col = chroma_client.create_collection("finqa_chunks")

t0 = time.time()
CHROMA_BATCH = 5000   # ChromaDB hard limit is 5461
for s in range(0, len(chunks), CHROMA_BATCH):
    e = s + CHROMA_BATCH
    chroma_col.add(
        ids=chunk_ids[s:e],
        embeddings=chunk_vectors[s:e].tolist(),
        documents=chunk_texts[s:e],
    )
chroma_build_time = time.time() - t0

t0 = time.time()
chroma_res = chroma_col.query(query_embeddings=question_vectors.tolist(), n_results=5)
chroma_search_time = time.time() - t0

def chroma_get_top5_texts(qi):
    return chroma_res["documents"][qi]   # ChromaDB returns the actual text back

chroma_recall = check_recall(chroma_get_top5_texts)
results["ChromaDB"] = {"build_time": chroma_build_time, "search_time": chroma_search_time, "recall": chroma_recall}
print(f"  build={chroma_build_time:.2f}s  search={chroma_search_time:.2f}s  recall={chroma_recall:.1%}")


# ==================================================================
# TOOL 3: Qdrant — local disk store, scales to server deployment
# ==================================================================
print("\n--- Testing Qdrant ---")

qclient = QdrantClient(path=str(ROOT / "data" / "processed" / "qdrant_db"))
try:
    qclient.delete_collection("finqa_chunks")
except Exception:
    pass
qclient.create_collection(
    collection_name="finqa_chunks",
    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
)

t0 = time.time()
qclient.upsert(
    collection_name="finqa_chunks",
    points=[
        PointStruct(id=i, vector=chunk_vectors[i].tolist(),
                    payload={"chunk_id": chunk_ids[i], "text": chunk_texts[i]})
        for i in range(len(chunks))
    ],
)
qdrant_build_time = time.time() - t0

t0 = time.time()
qdrant_res = [
    qclient.query_points(
        collection_name="finqa_chunks",
        query=question_vectors[i].tolist(),
        limit=5,
    ).points
    for i in range(len(eval_examples))
]
qdrant_search_time = time.time() - t0

def qdrant_get_top5_texts(qi):
    return [hit.payload["text"] for hit in qdrant_res[qi]]

qdrant_recall = check_recall(qdrant_get_top5_texts)
results["Qdrant"] = {"build_time": qdrant_build_time, "search_time": qdrant_search_time, "recall": qdrant_recall}
print(f"  build={qdrant_build_time:.2f}s  search={qdrant_search_time:.2f}s  recall={qdrant_recall:.1%}")

qclient.close()   # close before Python shutdown to silence the __del__ warning


# ==================================================================
# FINAL COMPARISON TABLE
# ==================================================================
print("\n--- FINAL COMPARISON ---")
print(f"{'Tool':<12}{'Build time':>12}{'Search time':>14}{'Recall@5':>10}")
for name, r in results.items():
    print(f"{name:<12}{r['build_time']:>11.2f}s{r['search_time']:>13.2f}s{r['recall']:>10.1%}")