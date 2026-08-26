"""Multi-query retrieval via LangChain MultiQueryRetriever."""

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore


def build_multi_query_retriever(vectorstore: VectorStore, llm: BaseLanguageModel, k: int = 5):
    base = vectorstore.as_retriever(search_kwargs={"k": k})
    return MultiQueryRetriever.from_llm(retriever=base, llm=llm)


def multi_query_search(retriever: MultiQueryRetriever, question: str) -> list[str]:
    docs = retriever.invoke(question)
    return [d.metadata["chunk_id"] for d in docs]
