"""Hybrid retrieval for FinQA: real BM25 (langchain_community, wraps rank_bm25)
+ dense (your existing Qdrant vectorstore), fused with real Reciprocal Rank
Fusion via LangChain's EnsembleRetriever (langchain_classic).

Why this matters specifically for FinQA, given what Project 4's HyDE result
showed: your questions already contain strong literal anchors (dates,
states, company terms) that overlap verbatim with the gold chunk text.
That's exactly what BM25 is built to exploit -- it should do well on
`control`-tagged questions (high literal overlap by definition) and should
struggle more than dense on `vocab`-tagged ones (low literal overlap by
definition, since BM25 needs term matches to score at all). This gives you
a real, data-grounded prediction to test, not just "try hybrid and see."
"""
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


def build_bm25_retriever(chunks: list[dict], k: int = 5) -> BM25Retriever:
    docs = [Document(page_content=c["text"], metadata={"chunk_id": c["chunk_id"], "doc_id": c["doc_id"]})
            for c in chunks]
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return bm25


def build_dense_retriever(vectorstore: VectorStore, k: int = 5):
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_hybrid_retriever(vectorstore: VectorStore, chunks: list[dict], k: int = 5,
                            bm25_weight: float = 0.5, dense_weight: float = 0.5) -> EnsembleRetriever:
    bm25 = build_bm25_retriever(chunks, k=k)
    dense = build_dense_retriever(vectorstore, k=k)
    return EnsembleRetriever(retrievers=[bm25, dense], weights=[bm25_weight, dense_weight])


def run_fn_from_retriever(retriever, k: int = 5):
    def run(query: str) -> list[str]:
        docs = retriever.invoke(query)
        seen, ids = set(), []
        for d in docs:
            cid = d.metadata["chunk_id"]
            if cid not in seen:
                seen.add(cid)
                ids.append(cid)
        return ids[:k]
    return run
