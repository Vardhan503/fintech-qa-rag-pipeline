"""Retrieval-only eval across every chunk store -> eval_results_*.json

Every strategy is scored by content overlap so the numbers are directly
comparable (a whole_table chunk can never equal a table_row gold id by id).
The row-level baseline is additionally scored by exact chunk_id match.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.retrieval_harness import (
    evaluate_chunking_strategy,
    evaluate_chunking_strategy_generic,
    gold_text_index,
    load_embedder,
    load_index_chunks,
)
from src.utils.io import load_jsonl, save_json

STRATEGIES = {
    "row_level": ("data/processed/chunks.jsonl",
                  "data/processed/eval_results_baseline_row_level.json"),
    "naive_fixed": ("data/processed/chunks_naive_fixed.jsonl",
                    "data/processed/eval_results_naive_fixed.json"),
    "whole_table": ("data/processed/chunks_whole_table.jsonl",
                    "data/processed/eval_results_whole_table.json"),
    "sentence_window": ("data/processed/chunks_sentence_window.jsonl",
                        "data/processed/eval_results_sentence_window.json"),
    "row_group_5": ("data/processed/chunks_grouped_5.jsonl",
                    "data/processed/eval_results_row_group_5.json"),
}


def main() -> None:
    model = load_embedder()
    eval_examples = load_jsonl("data/processed/eval_dataset.jsonl")
    chunk_id_to_gold_text = gold_text_index(load_jsonl("data/processed/chunks.jsonl"))
    print(f"{len(eval_examples)} questions, {len(chunk_id_to_gold_text)} gold-text references")

    for name, (chunks_path, out_path) in STRATEGIES.items():
        chunks = load_index_chunks(chunks_path)
        print(f"\nevaluating {name}: {len(chunks)} chunks...")
        summary = evaluate_chunking_strategy_generic(
            chunks, eval_examples, chunk_id_to_gold_text, model
        )
        for k, s in summary.items():
            print(f"  k={k:>2}  precision={s['precision']:.3f}  recall={s['recall']:.3f}")
        save_json(summary, out_path)

    print("\nexact-id scoring for the baseline (valid only within the row-level id scheme):")
    baseline = load_index_chunks("data/processed/chunks.jsonl")
    exact = evaluate_chunking_strategy(baseline, eval_examples, model)
    for k, s in exact.items():
        print(f"  k={k:>2}  precision={s['precision']:.3f}  recall={s['recall']:.3f}")
    save_json(exact, "data/processed/eval_results_row_level_exact_id.json")


if __name__ == "__main__":
    main()
