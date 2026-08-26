"""Hybrid BM25+dense (+ optional rerank) eval -> eval_results_hybrid_search.json

Uses an in-memory Qdrant index over row-level chunks. Reranker downloads
cross-encoder/ms-marco-MiniLM-L-6-v2 on first run.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.experiments import (
    build_memory_qdrant,
    evaluate_ndcg,
    make_tagged_sample,
    sample_composition,
)
from src.eval.retrieval_harness import load_index_chunks
from src.hybrid_search.hybrid import (
    build_bm25_retriever,
    build_dense_retriever,
    build_hybrid_retriever,
    run_fn_from_retriever,
)
from src.hybrid_search.reranker import build_reranker, rerank
from src.models import load_embedder
from src.utils.io import load_jsonl, save_json

OUT_PATH = "data/processed/eval_results_hybrid_search.json"
K = 10


def main() -> None:
    chunks = load_index_chunks("data/processed/chunks.jsonl")
    eval_examples = load_jsonl("data/processed/eval_dataset.jsonl")
    sample, gold_text_lookup = make_tagged_sample(
        eval_examples, chunks, per_op_n=10, seed=42
    )
    print(f"sample={len(sample)}  composition={sample_composition(sample)}")

    embeddings = load_embedder()
    print("indexing row-level chunks into in-memory Qdrant...")
    vectorstore = build_memory_qdrant(chunks, embeddings, "finqa_hybrid_search")

    results = {}

    dense = run_fn_from_retriever(build_dense_retriever(vectorstore, k=K), k=K)
    bm25 = run_fn_from_retriever(build_bm25_retriever(chunks, k=K), k=K)
    hybrid_5050 = run_fn_from_retriever(
        build_hybrid_retriever(vectorstore, chunks, k=K, bm25_weight=0.5, dense_weight=0.5),
        k=K,
    )
    hybrid_7030 = run_fn_from_retriever(
        build_hybrid_retriever(vectorstore, chunks, k=K, bm25_weight=0.7, dense_weight=0.3),
        k=K,
    )

    for name, fn in [
        ("dense_only", dense),
        ("bm25_only", bm25),
        ("hybrid_rrf_50_50", hybrid_5050),
        ("hybrid_rrf_70_30", hybrid_7030),
    ]:
        print(f"\n=== {name} ===")
        results[name] = evaluate_ndcg(fn, sample, k=K)
        print(json.dumps(results[name]["ALL"], indent=2))

    # Rerank the better of the two hybrid weightings on NDCG@ALL
    best_name = max(
        ("hybrid_rrf_50_50", "hybrid_rrf_70_30"),
        key=lambda n: results[n]["ALL"]["ndcg"],
    )
    best_fn = hybrid_5050 if best_name.endswith("50_50") else hybrid_7030
    print(f"\n=== hybrid_rerank (first stage={best_name}) ===")
    reranker = build_reranker()

    def hybrid_rerank_run(question, k=K):
        candidate_ids = best_fn(question)
        candidates = [
            {"chunk_id": cid, "text": gold_text_lookup.get(cid, "")}
            for cid in candidate_ids
        ]
        return rerank(question, candidates, reranker, top_k=k)

    results["hybrid_rerank"] = evaluate_ndcg(hybrid_rerank_run, sample, k=K)
    print(json.dumps(results["hybrid_rerank"]["ALL"], indent=2))

    results["sample_size"] = len(sample)
    results["sample_failure_mode_composition"] = sample_composition(sample)
    results["rerank_first_stage"] = best_name
    save_json(results, OUT_PATH)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
