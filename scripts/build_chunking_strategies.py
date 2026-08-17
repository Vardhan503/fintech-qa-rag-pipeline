"""Build the alternative chunk stores (strategies 2, 3, 4) from documents.jsonl.

Strategy 1 (row-level) is written by run_preprocessing.py, because
eval_dataset.jsonl's gold ids are derived from it.
"""

import sys
from pathlib import Path

# repo root on sys.path so `import src...` works both when this file is run
# directly and when it is imported (e.g. from a notebook)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import build_all
from src.chunking.naive_fixed import naive_fixed_size_chunks
from src.chunking.sentence_window import build_sentence_window_chunks
from src.chunking.whole_table import build_whole_table_chunks
from src.utils.io import load_jsonl, save_jsonl

STRATEGIES = {
    "naive_fixed": (naive_fixed_size_chunks, "data/processed/chunks_naive_fixed.jsonl"),
    "whole_table": (build_whole_table_chunks, "data/processed/chunks_whole_table.jsonl"),
    "sentence_window": (build_sentence_window_chunks, "data/processed/chunks_sentence_window.jsonl"),
}


def main() -> None:
    documents = load_jsonl("data/processed/documents.jsonl")
    print(f"loaded {len(documents)} documents")

    for name, (builder, out_path) in STRATEGIES.items():
        chunks = build_all(documents, builder)
        save_jsonl(chunks, out_path)
        print(f"{name:<16} {len(chunks):>6} chunks -> {out_path}")


if __name__ == "__main__":
    main()
