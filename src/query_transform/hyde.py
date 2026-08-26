"""HyDE: embed an LLM-generated hypothetical answer instead of the raw question."""

from langchain_classic.chains import HypotheticalDocumentEmbedder
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore

# Built-in fiqa prompt writes a narrative passage. FinQA gold rows look more like
# "Payments Volume is $2,457; Cards is 1,592." — so we also offer a terse variant.
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


def build_hyde_embedder(
    llm: BaseLanguageModel,
    base_embeddings: Embeddings,
    style: str = "row_level",
) -> HypotheticalDocumentEmbedder:
    if style == "fiqa":
        return HypotheticalDocumentEmbedder.from_llm(
            llm, base_embeddings, prompt_key="fiqa"
        )
    if style == "row_level":
        return HypotheticalDocumentEmbedder.from_llm(
            llm, base_embeddings, custom_prompt=ROW_LEVEL_HYDE_PROMPT
        )
    raise ValueError(f"unknown HyDE style {style!r}; expected 'fiqa' or 'row_level'")


def hyde_search(
    vectorstore: VectorStore,
    question: str,
    hyde_embedder: HypotheticalDocumentEmbedder,
    k: int = 5,
) -> list[str]:
    hyde_vector = hyde_embedder.embed_query(question)
    docs = vectorstore.similarity_search_by_vector(hyde_vector, k=k)
    return [d.metadata["chunk_id"] for d in docs]
