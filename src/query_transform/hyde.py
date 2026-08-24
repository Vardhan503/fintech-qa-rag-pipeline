"""HyDE for FinQA -- uses LangChain's own HypotheticalDocumentEmbedder, which
ships a prompt template literally tuned for financial QA ('fiqa'):

    "Please write a financial article passage to answer the question
     Question: {QUESTION}
     Passage:"

This is a good match for FinQA's own gold chunk text, which reads like
financial-report table/passage language ("Payments Volume (billions) is
$2,457...") -- exactly the register the fiqa prompt is designed to produce.

No hand-rolled prompt needed here; this is 100% library.
"""
from langchain_classic.chains import HypotheticalDocumentEmbedder
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore


def build_hyde_embedder(llm: BaseLanguageModel, base_embeddings: Embeddings) -> HypotheticalDocumentEmbedder:
    return HypotheticalDocumentEmbedder.from_llm(
        llm, base_embeddings, prompt_key="fiqa"
    )


def hyde_search(vectorstore: VectorStore, question: str,
                 hyde_embedder: HypotheticalDocumentEmbedder, k: int = 5) -> list[str]:
    """Generates a hypothetical financial passage, embeds it, searches by
    that vector directly (not by re-embedding raw question text)."""
    hyde_vector = hyde_embedder.embed_query(question)
    docs = vectorstore.similarity_search_by_vector(hyde_vector, k=k)
    return [d.metadata["chunk_id"] for d in docs]
