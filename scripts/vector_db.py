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
eval_examples    = load_jsonl("data/processed/eval_dataset.jsonl")

chunk_ids      = [c["chunk_id"] for c in chunks]
chunk_texts    = [c["text"]     for c in chunks]
question_texts = [ex["question"] for ex in eval_examples]

print(f"Loaded {len(chunks)} chunks and {len(eval_examples)} questions.")

# ---------- Step 2: embed once, reuse for all three tools ----------
# convert_to_tensor=False → sentence_transformers returns a plain numpy array directly,
# avoiding the PyTorch tensor → CPU → numpy view chain entirely.
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

# ---------- Step 3: ID-based Recall@5 ----------
# A question is "found" when at least one of the retrieved chunk_ids matches
# a gold_chunk_id. This is exact — no word-overlap approximation.
def check_recall(get_top5_ids_fn):
    found = 0
    scorable = 0
    for i, ex in enumerate(eval_examples):
        gold = set(ex["gold_chunk_ids"])
        if not gold:
            continue
        scorable += 1
        if gold & set(get_top5_ids_fn(i)):
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

def faiss_get_top5_ids(qi):
    return [chunk_ids[i] for i in faiss_indices[qi]]

faiss_recall = check_recall(faiss_get_top5_ids)
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

def chroma_get_top5_ids(qi):
    return chroma_res["ids"][qi]   # ChromaDB returns the IDs we inserted

chroma_recall = check_recall(chroma_get_top5_ids)
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
                    payload={"chunk_id": chunk_ids[i]})
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

def qdrant_get_top5_ids(qi):
    return [hit.payload["chunk_id"] for hit in qdrant_res[qi]]

qdrant_recall = check_recall(qdrant_get_top5_ids)
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
