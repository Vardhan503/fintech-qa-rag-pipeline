"""BM25 + dense hybrid retrieval via EnsembleRetriever."""

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


def build_bm25_retriever(chunks: list[dict], k: int = 5) -> BM25Retriever:
    docs = [
        Document(
            page_content=c["text"],
            metadata={"chunk_id": c["chunk_id"], "doc_id": c["doc_id"]},
        )
        for c in chunks
    ]
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return bm25


def build_dense_retriever(vectorstore: VectorStore, k: int = 5):
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_hybrid_retriever(
    vectorstore: VectorStore,
    chunks: list[dict],
    k: int = 5,
    bm25_weight: float = 0.5,
    dense_weight: float = 0.5,
) -> EnsembleRetriever:
    bm25 = build_bm25_retriever(chunks, k=k)
    dense = build_dense_retriever(vectorstore, k=k)
    return EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[bm25_weight, dense_weight],
    )


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
