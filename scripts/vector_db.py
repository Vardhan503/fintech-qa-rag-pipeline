"""Compare FAISS, Chroma, and Qdrant via LangChain vector stores.

Embeddings are computed once with HuggingFaceEmbeddings and reused so the
comparison is about the store, not the encoder.
"""

import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from src.eval.retrieval_harness import word_overlap_relevance
from src.models import load_embedder
from src.utils.io import load_jsonl, save_json

CHROMA_BATCH = 5000


def load_benchmark_data():
    chunks = [c for c in load_jsonl("data/processed/chunks_whole_table.jsonl")
              if not c.get("is_noise", False)]
    eval_examples = load_jsonl("data/processed/eval_dataset.jsonl")
    gold_text_lookup = {
        c["chunk_id"]: c["text"] for c in load_jsonl("data/processed/chunks.jsonl")
    }
    docs = [
        Document(page_content=c["text"], metadata={"chunk_id": c["chunk_id"]})
        for c in chunks
    ]
    return docs, eval_examples, gold_text_lookup


def recall_at_5(store, eval_examples, gold_text_lookup) -> float:
    found, scored = 0, 0
    for ex in eval_examples:
        gold_texts = [gold_text_lookup[g] for g in ex["gold_chunk_ids"] if g in gold_text_lookup]
        if not gold_texts:
            continue
        scored += 1
        retrieved = [d.page_content for d in store.similarity_search(ex["question"], k=5)]
        if any(word_overlap_relevance(rt, gt) for rt in retrieved for gt in gold_texts):
            found += 1
    return found / scored if scored else 0.0


def run_benchmark() -> dict:
    embeddings = load_embedder()
    docs, eval_examples, gold_text_lookup = load_benchmark_data()
    print(f"Loaded {len(docs)} chunks and {len(eval_examples)} questions.")
    results = {}

    print("\n--- Testing FAISS ---")
    t0 = time.time()
    faiss_store = FAISS.from_documents(docs, embeddings)
    build = time.time() - t0
    t0 = time.time()
    recall = recall_at_5(faiss_store, eval_examples, gold_text_lookup)
    search = time.time() - t0
    results["FAISS"] = {"build_time": build, "search_time": search, "recall": recall}
    print(f"  build={build:.2f}s  search={search:.2f}s  recall={recall:.1%}")

    print("\n--- Testing ChromaDB ---")
    chroma_dir = ROOT / "data" / "processed" / "chroma_db"
    t0 = time.time()
    chroma_store = Chroma(
        collection_name="finqa_chunks",
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )
    for start in range(0, len(docs), CHROMA_BATCH):
        chroma_store.add_documents(docs[start:start + CHROMA_BATCH])
    build = time.time() - t0
    t0 = time.time()
    recall = recall_at_5(chroma_store, eval_examples, gold_text_lookup)
    search = time.time() - t0
    results["ChromaDB"] = {"build_time": build, "search_time": search, "recall": recall}
    print(f"  build={build:.2f}s  search={search:.2f}s  recall={recall:.1%}")

    print("\n--- Testing Qdrant ---")
    qpath = ROOT / "data" / "processed" / "qdrant_db"
    client = QdrantClient(path=str(qpath))
    try:
        client.delete_collection("finqa_chunks")
    except Exception:
        pass
    dim = len(embeddings.embed_query("dimension probe"))
    client.create_collection(
        collection_name="finqa_chunks",
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    qdrant_store = QdrantVectorStore(
        client=client,
        collection_name="finqa_chunks",
        embedding=embeddings,
    )
    t0 = time.time()
    qdrant_store.add_documents(docs, batch_size=256)
    build = time.time() - t0
    t0 = time.time()
    recall = recall_at_5(qdrant_store, eval_examples, gold_text_lookup)
    search = time.time() - t0
    results["Qdrant"] = {"build_time": build, "search_time": search, "recall": recall}
    print(f"  build={build:.2f}s  search={search:.2f}s  recall={recall:.1%}")
    client.close()

    print("\n--- FINAL COMPARISON ---")
    print(f"{'Tool':<12}{'Build time':>12}{'Search time':>14}{'Recall@5':>10}")
    for name, r in results.items():
        print(f"{name:<12}{r['build_time']:>11.2f}s{r['search_time']:>13.2f}s{r['recall']:>10.1%}")

    out = "data/processed/eval_results_vector_db.json"
    save_json(results, out)
    print(f"\nwrote {out}")
    return results


if __name__ == "__main__":
    run_benchmark()
