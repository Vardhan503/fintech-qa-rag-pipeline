"""Multi-Query retrieval for FinQA -- uses LangChain's own MultiQueryRetriever
(no hand-rolled paraphrase generation). It generates several reworded
versions of the query with the LLM, retrieves for each, and returns the
UNION of unique documents (deduplicated, not rank-fused -- that's the
library's own default behavior, different from the RRF-fused version we
hand-built for the Nimbus corpus in Project 4).
"""
import logging
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore

# MultiQueryRetriever logs each generated query at INFO level by default --
# useful the first time you run this (to sanity-check what it's generating),
# noisy for a full eval loop over dozens of questions. Uncomment to inspect:
# logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


def build_multi_query_retriever(vectorstore: VectorStore, llm: BaseLanguageModel, k: int = 5):
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)


def multi_query_search(retriever: MultiQueryRetriever, question: str) -> list[str]:
    docs = retriever.invoke(question)
    return [d.metadata["chunk_id"] for d in docs]
