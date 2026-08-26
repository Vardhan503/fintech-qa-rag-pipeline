"""Decompose compound questions into sub-queries, retrieve each, then RRF."""

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field

from src.retrieval.fusion import reciprocal_rank_fusion


class SubQuestions(BaseModel):
    is_compound: bool = Field(
        description="True if the question needs 2+ independent facts combined"
    )
    sub_questions: list[str] = Field(
        description=(
            "If compound: 2-3 independent sub-questions. "
            "Otherwise: a one-item list with the original question."
        )
    )


DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You analyze financial questions about company reports. Decide if the "
        "question needs TWO OR MORE independently retrievable figures (e.g. a "
        "ratio or year-over-year change). If so, split into that many "
        "sub-questions. If it asks for a single figure, return it unchanged.",
    ),
    ("human", "{question}"),
])


def build_decompose_chain(llm: BaseLanguageModel):
    return DECOMPOSE_PROMPT | llm.with_structured_output(SubQuestions)


def decompose_search(vectorstore: VectorStore, question: str, decompose_chain, k: int = 5) -> list[str]:
    result: SubQuestions = decompose_chain.invoke({"question": question})
    sub_qs = result.sub_questions or [question]
    ranked = []
    for sq in sub_qs:
        ranked.append([d.metadata["chunk_id"] for d in vectorstore.similarity_search(sq, k=k)])
    return reciprocal_rank_fusion(ranked)[:k]
