"""Build the RAG index and ask one or more questions.

Requires Ollama running locally with LLM_MODEL (default qwen2.5:7b-instruct).

Examples:
  python scripts/run_rag.py --setup
  python scripts/run_rag.py --ask "what is the average payment volume per transaction for american express?"
  python scripts/run_rag.py --setup --ask "your question here"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.pipeline import SimpleRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="FinQA simple RAG (whole-table + Qdrant + Ollama)")
    parser.add_argument("--setup", action="store_true", help="(re)build the Qdrant index from chunks")
    parser.add_argument("--ask", type=str, help="question to answer")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--chunks",
        default="data/processed/chunks_whole_table.jsonl",
        help="chunk store to index",
    )
    args = parser.parse_args()

    if not args.setup and not args.ask:
        parser.error("pass --setup and/or --ask")

    rag = SimpleRAG()
    try:
        rag.setup(chunks_path=args.chunks, rebuild=args.setup)
        if args.ask:
            result = rag.ask(args.ask, top_k=args.top_k)
            print("\nQuestion:", result["question"])
            print("Answer:", result["answer"])
            print("\nSources used:")
            for i, src in enumerate(result["sources"], 1):
                print(f"  [{i}] {src[:120]}{'...' if len(src) > 120 else ''}")
    finally:
        rag.close()


if __name__ == "__main__":
    main()
