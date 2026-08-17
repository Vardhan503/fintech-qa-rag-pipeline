"""raw dev.json -> documents.jsonl -> chunks.jsonl -> eval_dataset.jsonl

Wiring only; the logic lives in src/data and src/chunking.
"""

import sys
from pathlib import Path

# repo root on sys.path so `import src...` works both when this file is run
# directly and when it is imported (e.g. from a notebook)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import build_all
from src.chunking.row_level import build_row_level_chunks
from src.data.loader import load_raw_finqa
from src.data.reconstruction import build_documents
from src.eval.gold_mapping import build_eval_dataset
from src.utils.io import save_jsonl


def main() -> None:
    data = load_raw_finqa()
    print(f"loaded {len(data)} raw QA examples")

    documents = build_documents(data)
    save_jsonl(documents, "data/processed/documents.jsonl")
    print(f"reconstructed {len(documents)} unique documents -> documents.jsonl")

    chunks = build_all(documents, build_row_level_chunks)
    save_jsonl(chunks, "data/processed/chunks.jsonl")
    n_noise = sum(c["is_noise"] for c in chunks)
    print(f"built {len(chunks)} row-level chunks ({n_noise} flagged noise) -> chunks.jsonl")

    eval_examples, missing = build_eval_dataset(data, documents, chunks)
    save_jsonl(eval_examples, "data/processed/eval_dataset.jsonl")
    print(f"eval_dataset: {len(eval_examples)} examples, {missing} unresolved gold references")

    if missing:
        raise SystemExit(
            f"{missing} gold references do not resolve to a chunk_id -- index alignment is broken"
        )


if __name__ == "__main__":
    main()
