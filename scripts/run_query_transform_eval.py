"""Query-transform retrieval eval -> data/processed/eval_results_query_transform.json

Needs Ollama (qwen2.5:7b-instruct). Uses an in-memory Qdrant index over
row-level chunks and the same stratified sample as other retrieval experiments.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.experiments import (
    build_memory_qdrant,
    evaluate_run,
    make_tagged_sample,
    sample_composition,
)
from src.eval.retrieval_harness import load_index_chunks
from src.models import load_embedder, load_llm
from src.query_transform.decompose import build_decompose_chain, decompose_search
from src.query_transform.hyde import build_hyde_embedder, hyde_search
from src.query_transform.multi_query import build_multi_query_retriever, multi_query_search
from src.query_transform.step_back import build_step_back_chain, step_back_search
from src.utils.io import load_jsonl, save_json

OUT_PATH = "data/processed/eval_results_query_transform.json"
K = 5


def main() -> None:
    chunks = load_index_chunks("data/processed/chunks.jsonl")
    eval_examples = load_jsonl("data/processed/eval_dataset.jsonl")
    sample, _ = make_tagged_sample(eval_examples, chunks, per_op_n=10, seed=42)
    print(f"sample={len(sample)}  composition={sample_composition(sample)}")

    embeddings = load_embedder()
    llm = load_llm()
    print("indexing row-level chunks into in-memory Qdrant...")
    vectorstore = build_memory_qdrant(chunks, embeddings, "finqa_query_transform")

    results = {}

    def baseline_run(question, k=K):
        docs = vectorstore.similarity_search(question, k=k)
        return [d.metadata["chunk_id"] for d in docs]

    print("\n=== baseline ===")
    results["baseline"], _ = evaluate_run(baseline_run, sample, k=K)
    print(json.dumps(results["baseline"]["ALL"], indent=2))

    print("\n=== hyde (row_level) ===")
    hyde = build_hyde_embedder(llm, embeddings, style="row_level")

    def hyde_run(question, k=K):
        return hyde_search(vectorstore, question, hyde, k=k)

    results["hyde"], _ = evaluate_run(hyde_run, sample, k=K)
    print(json.dumps(results["hyde"]["ALL"], indent=2))

    print("\n=== multi_query ===")
    mq = build_multi_query_retriever(vectorstore, llm, k=K)

    def mq_run(question, k=K):
        return multi_query_search(mq, question)[:k]

    results["multi_query"], _ = evaluate_run(mq_run, sample, k=K)
    print(json.dumps(results["multi_query"]["ALL"], indent=2))

    print("\n=== step_back ===")
    sb = build_step_back_chain(llm)

    def sb_run(question, k=K):
        return step_back_search(vectorstore, question, sb, k=k)

    results["step_back"], _ = evaluate_run(sb_run, sample, k=K)
    print(json.dumps(results["step_back"]["ALL"], indent=2))

    print("\n=== decompose ===")
    dec = build_decompose_chain(llm)

    def dec_run(question, k=K):
        return decompose_search(vectorstore, question, dec, k=k)

    results["decompose"], _ = evaluate_run(dec_run, sample, k=K)
    print(json.dumps(results["decompose"]["ALL"], indent=2))

    results["sample_size"] = len(sample)
    results["sample_failure_mode_composition"] = sample_composition(sample)
    save_json(results, OUT_PATH)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
