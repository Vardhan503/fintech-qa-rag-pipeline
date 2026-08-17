"""Sweep table chunk granularity (rows per chunk) -> eval_results_chunk_size.json

Retrieval only, so no Ollama needed. rows_per_group=1 is the row-level baseline
and 999 is effectively whole-table, with the interesting question being whether
anything in between beats both ends.
"""

import sys
from pathlib import Path

# repo root on sys.path so `import src...` works both when this file is run
# directly and when it is imported (e.g. from a notebook)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import build_all
from src.chunking.row_group import build_row_group_chunks
from src.eval.retrieval_harness import gold_text_index, load_embedder, recall_at_k_by_content
from src.utils.io import load_jsonl, save_json

SIZES = [1, 2, 3, 5, 999]
TOP_K = 5


def label_for(size: int) -> str:
    return "whole table" if size >= 999 else f"{size} row(s) per chunk"


def run_sweep(sizes=SIZES, top_k: int = TOP_K, model=None) -> dict:
    model = model or load_embedder()
    documents = load_jsonl("data/processed/documents.jsonl")
    eval_examples = load_jsonl("data/processed/eval_dataset.jsonl")
    gold_text_lookup = gold_text_index(load_jsonl("data/processed/chunks.jsonl"))

    results = {}
    for size in sizes:
        chunks = build_all(documents, lambda doc, s=size: build_row_group_chunks(doc, s))
        scores = recall_at_k_by_content(chunks, eval_examples, gold_text_lookup, model, k=top_k)
        results[label_for(size)] = {"rows_per_group": size, **scores}
        print(f"{label_for(size):<22} recall@{top_k} = {scores['recall']:.1%}   "
              f"({scores['n_chunks']} chunks, {scores['n_scored']} questions scored)")
    return results


def main() -> None:
    print(f"sweeping table chunk size over {SIZES}, top_k={TOP_K}\n")
    results = run_sweep()

    print("\n--- FINAL COMPARISON ---")
    for label, r in sorted(results.items(), key=lambda kv: -kv[1]["recall"]):
        print(f"{label:<22} recall={r['recall']:.1%}  chunks={r['n_chunks']}")

    save_json(results, "data/processed/eval_results_chunk_size.json")


if __name__ == "__main__":
    main()
