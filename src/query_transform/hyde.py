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
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore

# The library's built-in "fiqa" prompt asks for a narrative "financial
# article passage" -- a style mismatch against this corpus's actual gold
# chunks, which are terse table-row facts ("Payments Volume (billions) is
# $2,457; ..."). This custom prompt asks for that same terse style instead.
# A/B this against prompt_key="fiqa" below once diagnostic_hyde_inspect.py
# confirms the style-mismatch hypothesis.
ROW_LEVEL_HYDE_PROMPT = PromptTemplate(
    input_variables=["QUESTION"],
    template=(
        "Write ONE short, dense sentence in the exact style of a financial "
        "report table row that would contain the answer to this question. "
        "Use the pattern 'Metric name is value; Metric name is value;' with "
        "plausible-sounding company/metric names and numbers. Do NOT write "
        "a paragraph or explanation -- output only the single row-style "
        "sentence.\n\nQuestion: {QUESTION}\nRow:"
    ),
)


def build_hyde_embedder(llm: BaseLanguageModel, base_embeddings: Embeddings,
                         style: str = "row_level") -> HypotheticalDocumentEmbedder:
    """style='fiqa' uses LangChain's financial-article prompt;
    style='row_level' uses a terse table-row prompt matched to FinQA gold text.
    """
    if style == "fiqa":
        return HypotheticalDocumentEmbedder.from_llm(llm, base_embeddings, prompt_key="fiqa")
    if style == "row_level":
        return HypotheticalDocumentEmbedder.from_llm(
            llm, base_embeddings, custom_prompt=ROW_LEVEL_HYDE_PROMPT
        )
    raise ValueError(f"unknown HyDE style {style!r}; expected 'fiqa' or 'row_level'")


def hyde_search(vectorstore: VectorStore, question: str,
                 hyde_embedder: HypotheticalDocumentEmbedder, k: int = 5) -> list[str]:
    """Generates a hypothetical financial passage, embeds it, searches by
    that vector directly (not by re-embedding raw question text)."""
    hyde_vector = hyde_embedder.embed_query(question)
    docs = vectorstore.similarity_search_by_vector(hyde_vector, k=k)
    return [d.metadata["chunk_id"] for d in docs]