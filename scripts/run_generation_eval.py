"""End-to-end generation eval across strategies -> eval_results_generation.json

Requires a local Ollama serving LLM_MODEL. Roughly 5 strategies x 44 questions
of sequential LLM calls, so budget ~15-20 minutes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.generation_harness import LLM_MODEL, run_generation_eval
from src.eval.retrieval_harness import gold_text_index, load_embedder, load_index_chunks
from src.eval.sampling import describe_sample, stratified_sample_by_op
from src.utils.io import PROCESSED_DIR, load_jsonl, save_json, save_jsonl

SAMPLE_PATH = "data/processed/gen_eval_sample.jsonl"

STRATEGIES = {
    "row_level":       "data/processed/chunks.jsonl",
    "naive_fixed":     "data/processed/chunks_naive_fixed.jsonl",
    "whole_table":     "data/processed/chunks_whole_table.jsonl",
    "sentence_window": "data/processed/chunks_sentence_window.jsonl",
    "row_group_5":     "data/processed/chunks_grouped_5.jsonl",
}


def get_sample(per_op_n: int = 5) -> list[dict]:
    """Reuse the sample on disk if present so every strategy is graded on
    exactly the same questions."""
    if (PROCESSED_DIR / "gen_eval_sample.jsonl").exists():
        sample = load_jsonl(SAMPLE_PATH)
        print(f"reusing existing sample: {len(sample)} questions")
        return sample

    eval_examples = load_jsonl("data/processed/eval_dataset.jsonl")
    sample = stratified_sample_by_op(eval_examples, per_op_n=per_op_n)
    for op, n_sampled, n_available in describe_sample(eval_examples, per_op_n=per_op_n):
        print(f"  {op:<15} {n_sampled}/{n_available} sampled")
    save_jsonl(sample, SAMPLE_PATH)
    print(f"stratified sample: {len(sample)} questions -> {SAMPLE_PATH}")
    return sample


def main() -> None:
    model = load_embedder()
    sample = get_sample()
    gold = gold_text_index(load_jsonl("data/processed/chunks.jsonl"))
    print(f"generation eval with LLM={LLM_MODEL}")

    all_results = {}
    for name, chunks_path in STRATEGIES.items():
        print(f"\n=== {name} ===")
        chunks = load_index_chunks(chunks_path)
        result = run_generation_eval(
            chunks, sample, model, top_k=5, gold_text_by_id=gold
        )
        n_correct = sum(r["correct"] for r in result["results"])
        print(f"{name} accuracy: {result['accuracy']:.1%}  "
              f"({n_correct}/{result['n']})")
        all_results[name] = result

    print("\n=== FINAL COMPARISON ===")
    for name, r in sorted(all_results.items(), key=lambda kv: -kv[1]["accuracy"]):
        print(f"{name:<18} accuracy={r['accuracy']:.1%}")

    save_json({k: v["accuracy"] for k, v in all_results.items()},
              "data/processed/generation_comparison.json")
    save_json(all_results, "data/processed/eval_results_generation.json")


if __name__ == "__main__":
    main()
