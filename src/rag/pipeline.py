"""RAG pipeline: retrieve whole-table chunks from Qdrant, answer with ChatOllama."""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from src.eval.generation_harness import extract_final_answer
from src.models import LLM_MODEL, load_embedder, load_llm
from src.utils.io import PROCESSED_DIR, load_jsonl, resolve

COLLECTION_NAME = "finqa_chunks"
DEFAULT_CHUNKS_PATH = "data/processed/chunks_whole_table.jsonl"
QDRANT_PATH = PROCESSED_DIR / "qdrant_db_final"

PROMPT = ChatPromptTemplate.from_template(
    """You are a financial analyst. Answer using ONLY the context below.
Keep your answer short. Finish with a line exactly like this: ANSWER: <number or short answer>

Context:
{context}

Question: {question}
"""
)


def _format_docs(docs: list[Document]) -> str:
    return "\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))


class SimpleRAG:
    """Build a LangChain retriever once with setup(), then call ask(question)."""

    def __init__(self, qdrant_path=None, llm_model: str = LLM_MODEL):
        self.llm_model = llm_model
        self.qdrant_path = resolve(qdrant_path or QDRANT_PATH)
        print("Loading embedding model...")
        self.embeddings = load_embedder()
        self.llm = load_llm(llm_model)
        self.client = QdrantClient(path=str(self.qdrant_path))
        self.store = None
        self.chain = None

    def setup(self, chunks_path: str = DEFAULT_CHUNKS_PATH, rebuild: bool = True) -> None:
        chunks = [c for c in load_jsonl(chunks_path) if not c.get("is_noise", False)]
        docs = [
            Document(
                page_content=c["text"],
                metadata={"chunk_id": c["chunk_id"], "doc_id": c["doc_id"]},
            )
            for c in chunks
        ]

        if rebuild:
            print(f"Indexing {len(docs)} chunks into Qdrant...")
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            dim = len(self.embeddings.embed_query("dimension probe"))
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            self.store = QdrantVectorStore(
                client=self.client,
                collection_name=COLLECTION_NAME,
                embedding=self.embeddings,
            )
            self.store.add_documents(docs, batch_size=256)
        else:
            self.store = QdrantVectorStore(
                client=self.client,
                collection_name=COLLECTION_NAME,
                embedding=self.embeddings,
            )

        retriever = self.store.as_retriever(search_kwargs={"k": 5})
        self.chain = (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | PROMPT
            | self.llm
            | StrOutputParser()
        )
        print(f"Setup done — index at {self.qdrant_path}")

    def search(self, question: str, top_k: int = 5) -> list[dict]:
        if self.store is None:
            raise RuntimeError("Call setup() before search().")
        docs = self.store.similarity_search(question, k=top_k)
        return [{"text": d.page_content, **d.metadata} for d in docs]

    def ask(self, question: str, top_k: int = 5) -> dict:
        if self.chain is None:
            raise RuntimeError("Call setup() before ask().")

        matches = self.search(question, top_k=top_k)
        try:
            raw = self.chain.invoke(question)
        except Exception as exc:
            raw = ""
            print(f"LLM call failed: {exc}")

        return {
            "question": question,
            "answer": extract_final_answer(raw),
            "raw_ai_response": raw,
            "sources": [m["text"] for m in matches],
            "chunk_ids": [m.get("chunk_id") for m in matches],
            "doc_ids": [m.get("doc_id") for m in matches],
        }

    def close(self) -> None:
        self.client.close()
