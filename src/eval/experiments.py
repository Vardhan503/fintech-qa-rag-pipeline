"""Shared helpers for query-transform and hybrid-search evals."""

from __future__ import annotations

import time
from collections import Counter, defaultdict

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from src.eval.retrieval_harness import context_precision_at_k, context_recall_at_k
from src.eval.sampling import stratified_sample_by_op
from src.hybrid_search.ndcg import ndcg_at_k


def word_overlap_ratio(a: str, b: str) -> float:
    a_words = {w.lower().strip(".,;:()%$") for w in a.split()}
    b_words = {w.lower().strip(".,;:()%$") for w in b.split()}
    if not a_words:
        return 0.0
    return len(a_words & b_words) / len(a_words)


def tag_failure_mode(ex: dict, gold_text_lookup: dict) -> str:
    """multihop = 2+ gold ids; vocab = low Q↔gold lexical overlap; else control."""
    gold_ids = ex["gold_chunk_ids"]
    if len(gold_ids) >= 2:
        return "multihop"
    gold_texts = [gold_text_lookup[g] for g in gold_ids if g in gold_text_lookup]
    if gold_texts and max(word_overlap_ratio(ex["question"], gt) for gt in gold_texts) < 0.15:
        return "vocab"
    return "control"


def make_tagged_sample(eval_examples: list[dict], chunks: list[dict],
                       per_op_n: int = 10, seed: int = 42) -> tuple[list[dict], dict[str, str]]:
    gold_text_lookup = {c["chunk_id"]: c["text"] for c in chunks}
    sample = stratified_sample_by_op(eval_examples, per_op_n=per_op_n, seed=seed)
    sample = [ex for ex in sample if ex["gold_chunk_ids"]]
    for ex in sample:
        ex["failure_mode"] = tag_failure_mode(ex, gold_text_lookup)
    return sample, gold_text_lookup


def build_memory_qdrant(chunks: list[dict], embeddings, collection_name: str) -> QdrantVectorStore:
    docs = [
        Document(
            page_content=c["text"],
            metadata={"chunk_id": c["chunk_id"], "doc_id": c["doc_id"]},
        )
        for c in chunks
    ]
    client = QdrantClient(":memory:")
    dim = len(embeddings.embed_query("dimension probe"))
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    store.add_documents(docs, batch_size=256)
    return store


def evaluate_run(run_fn, sample: list[dict], k: int = 5) -> tuple[dict, list[dict]]:
    """Exact-id recall/precision @ k, broken out by failure_mode."""
    per_mode: dict[str, list] = defaultdict(list)
    rows = []
    for ex in sample:
        t0 = time.time()
        retrieved = run_fn(ex["question"])
        latency = time.time() - t0
        gold_ids = set(ex["gold_chunk_ids"])
        recall = context_recall_at_k(retrieved, gold_ids, k)
        precision = context_precision_at_k(retrieved, gold_ids, k)
        if recall is None:
            continue
        per_mode[ex["failure_mode"]].append((recall, precision, latency))
        rows.append({
            "id": ex["id"],
            "question": ex["question"],
            "mode": ex["failure_mode"],
            "recall": recall,
            "precision": precision,
            "latency_s": latency,
        })

    summary = {}
    for mode, vals in per_mode.items():
        n = len(vals)
        summary[mode] = {
            "n": n,
            "recall": sum(v[0] for v in vals) / n,
            "precision": sum(v[1] for v in vals) / n,
            "avg_latency_s": sum(v[2] for v in vals) / n,
        }
    all_vals = [v for vals in per_mode.values() for v in vals]
    n = len(all_vals)
    summary["ALL"] = {
        "n": n,
        "recall": sum(v[0] for v in all_vals) / n if n else 0.0,
        "precision": sum(v[1] for v in all_vals) / n if n else 0.0,
        "avg_latency_s": sum(v[2] for v in all_vals) / n if n else 0.0,
    }
    return summary, rows


def evaluate_ndcg(run_fn, sample: list[dict], k: int = 10) -> dict:
    """NDCG@k by failure_mode (binary relevance from gold ids)."""
    per_mode: dict[str, list] = defaultdict(list)
    latencies: dict[str, list] = defaultdict(list)
    for ex in sample:
        t0 = time.time()
        retrieved = run_fn(ex["question"])
        latencies[ex["failure_mode"]].append(time.time() - t0)
        per_mode[ex["failure_mode"]].append(
            ndcg_at_k(retrieved, set(ex["gold_chunk_ids"]), k)
        )

    summary = {}
    for mode, scores in per_mode.items():
        summary[mode] = {
            "n": len(scores),
            "ndcg": sum(scores) / len(scores),
            "avg_latency_s": sum(latencies[mode]) / len(latencies[mode]),
        }
    all_scores = [s for scores in per_mode.values() for s in scores]
    all_lat = [t for times in latencies.values() for t in times]
    summary["ALL"] = {
        "n": len(all_scores),
        "ndcg": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "avg_latency_s": sum(all_lat) / len(all_lat) if all_lat else 0.0,
    }
    return summary


def sample_composition(sample: list[dict]) -> dict[str, int]:
    return dict(Counter(ex["failure_mode"] for ex in sample))
