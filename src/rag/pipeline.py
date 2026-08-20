"""Simple RAG pipeline: question in -> Qdrant search -> Ollama -> answer out.

Uses whole-table chunks (best generation accuracy from the chunking lab) and
persists the vector index under data/processed/qdrant_db_final/.
"""

import os

# macOS arm64: avoid tokenizer / BLAS threading races during encode
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from src.eval.generation_harness import LLM_MODEL, build_prompt, call_llm, extract_final_answer
from src.eval.retrieval_harness import MODEL_NAME, load_embedder
from src.utils.io import PROCESSED_DIR, load_jsonl, resolve

COLLECTION_NAME = "finqa_chunks"
DEFAULT_CHUNKS_PATH = "data/processed/chunks_whole_table.jsonl"
QDRANT_PATH = PROCESSED_DIR / "qdrant_db_final"


class SimpleRAG:
    """Build the Qdrant index once with setup(), then call ask(question) repeatedly."""

    def __init__(
        self,
        qdrant_path=None,
        embed_model_name: str = MODEL_NAME,
        llm_model: str = LLM_MODEL,
    ):
        self.llm_model = llm_model
        self.qdrant_path = resolve(qdrant_path or QDRANT_PATH)
        print("Loading embedding model...")
        self.embed_model = load_embedder(embed_model_name)
        self.qdrant = QdrantClient(path=str(self.qdrant_path))
        self.is_ready = False

    def setup(self, chunks_path: str = DEFAULT_CHUNKS_PATH, rebuild: bool = True) -> None:
        """Embed chunks and upsert into Qdrant. Run once (or when chunks change)."""
        chunks = [c for c in load_jsonl(chunks_path) if not c.get("is_noise", False)]
        chunk_ids = [c["chunk_id"] for c in chunks]
        chunk_texts = [c["text"] for c in chunks]

        print(f"Embedding {len(chunks)} chunks...")
        vectors = np.asarray(
            self.embed_model.encode(
                chunk_texts,
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=True,
                convert_to_tensor=False,
            ),
            dtype=np.float32,
        ).copy()
        dimension = vectors.shape[1]

        if rebuild:
            print("Building search index...")
            try:
                self.qdrant.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            self.qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=i,
                        vector=vectors[i].tolist(),
                        payload={
                            "chunk_id": chunk_ids[i],
                            "text": chunk_texts[i],
                            "doc_id": chunks[i]["doc_id"],
                        },
                    )
                    for i in range(len(chunks))
                ],
            )
        self.is_ready = True
        print(f"Setup done — index at {self.qdrant_path}")

    def search(self, question: str, top_k: int = 5) -> list[dict]:
        """Return top-k chunk payloads for a question."""
        if not self.is_ready:
            raise RuntimeError("Call setup() before search().")

        qvec = np.asarray(
            self.embed_model.encode(
                [question],
                normalize_embeddings=True,
                convert_to_tensor=False,
            ),
            dtype=np.float32,
        )[0]

        hits = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=qvec.tolist(),
            limit=top_k,
        ).points
        return [hit.payload for hit in hits]

    def ask(self, question: str, top_k: int = 5) -> dict:
        """Retrieve context, call the local LLM, return answer + sources."""
        if not self.is_ready:
            raise RuntimeError("Call setup() before ask().")

        matches = self.search(question, top_k=top_k)
        context_pieces = [m["text"] for m in matches]

        try:
            raw = call_llm(build_prompt(question, context_pieces), llm_model=self.llm_model)
        except Exception as exc:
            raw = ""
            print(f"LLM call failed: {exc}")

        return {
            "question": question,
            "answer": extract_final_answer(raw),
            "raw_ai_response": raw,
            "sources": context_pieces,
            "chunk_ids": [m.get("chunk_id") for m in matches],
            "doc_ids": [m.get("doc_id") for m in matches],
        }

    def close(self) -> None:
        self.qdrant.close()
