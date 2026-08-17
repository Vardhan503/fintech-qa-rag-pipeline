"""End-to-end generation eval across strategies -> eval_results_generation.json

Requires a local Ollama serving LLM_MODEL. Roughly 6 strategies x 44 questions
of sequential LLM calls, so budget 20-25 minutes.
Parent-child (strategy 6) is also run and saved to eval_results_parent_child.json.
"""

import sys
from pathlib import Path

# repo root on sys.path so `import src...` works both when this file is run
# directly and when it is imported (e.g. from a notebook)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking.parent_child import build_index_chunks, build_parent_text_map
from src.eval.generation_harness import LLM_MODEL, run_generation_eval, run_parent_child_eval
from src.eval.retrieval_harness import load_embedder, load_index_chunks
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
    """Reuse the sample on disk if present, so every strategy -- including runs
    from previous sessions -- is graded on exactly the same questions."""
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
    documents = load_jsonl("data/processed/documents.jsonl")
    model = load_embedder()
    sample = get_sample()
    print(f"generation eval with LLM={LLM_MODEL}")

    # ── strategies 1-5 (standard loop) ──────────────────────────────────────
    all_results = {}
    for name, chunks_path in STRATEGIES.items():
        print(f"\n=== {name} ===")
        chunks = load_index_chunks(chunks_path)
        result = run_generation_eval(chunks, sample, model, top_k=5)
        print(f"{name} accuracy: {result['accuracy']:.1%}  "
              f"({sum(r['correct'] for r in result['results'])}/{result['n']})")
        all_results[name] = result

    # ── strategy 6: parent-child ─────────────────────────────────────────────
    print("\n=== parent_child ===")
    pc_chunks = [c for c in build_index_chunks(documents) if not c.get("is_noise")]
    parent_map = build_parent_text_map(documents)
    pc_result = run_parent_child_eval(pc_chunks, parent_map, sample, model, top_k=5)
    print(f"parent_child accuracy: {pc_result['accuracy']:.1%}  "
          f"({sum(r['correct'] for r in pc_result['results'])}/{pc_result['n']})")
    all_results["parent_child"] = pc_result
    save_json(pc_result, "data/processed/eval_results_parent_child.json")

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n=== FINAL COMPARISON ===")
    for name, r in sorted(all_results.items(), key=lambda kv: -kv[1]["accuracy"]):
        print(f"{name:<18} accuracy={r['accuracy']:.1%}")

    save_json({k: v["accuracy"] for k, v in all_results.items()},
              "data/processed/generation_comparison.json")
    save_json(all_results, "data/processed/eval_results_generation.json")


if __name__ == "__main__":
    main()
