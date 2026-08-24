"""Sub-Query Decomposition for FinQA.

FinQA is a great real-world test of this technique: 396 of 883 questions
(45%) have 2+ gold_chunk_ids -- genuinely compound, multi-hop questions by
construction (the FinQA `program` field literally chains multiple table
values together, e.g. divide(637, const_5) needs the $637 figure and knows
to divide by a constant -- but two-operand programs like
subtract(193.5, const_100) that pull TWO different reported figures are
exactly where decomposition should help retrieval).

Uses `llm.with_structured_output(Pydantic model)` -- a real library feature
(function-calling / JSON-mode under the hood), not string-parsing an LLM's
free-text list like a from-scratch version would need to.
"""
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore

from src.query_transform.fusion import reciprocal_rank_fusion


class SubQuestions(BaseModel):
    is_compound: bool = Field(description="True if the question needs 2+ independent facts combined")
    sub_questions: list[str] = Field(
        description="If is_compound, 2-3 independent sub-questions. If not, a single-item list "
                    "containing the original question unchanged.")


DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You analyze financial questions asked against a company's financial "
     "report tables. Decide if the question requires combining TWO OR MORE "
     "independently-retrievable figures (e.g. a percentage change between "
     "two years, a ratio between two different line items). If so, split it "
     "into that many independent sub-questions, each retrievable on its own. "
     "If the question already asks for a single figure, return it unchanged "
     "as a single-item list."),
    ("human", "{question}"),
])


def build_decompose_chain(llm: BaseLanguageModel):
    structured_llm = llm.with_structured_output(SubQuestions)
    return DECOMPOSE_PROMPT | structured_llm


def decompose_search(vectorstore: VectorStore, question: str, decompose_chain, k: int = 5) -> list[str]:
    result: SubQuestions = decompose_chain.invoke({"question": question})
    sub_qs = result.sub_questions if result.sub_questions else [question]
    ranked_lists = []
    for sq in sub_qs:
        docs = vectorstore.similarity_search(sq, k=k)
        ranked_lists.append([d.metadata["chunk_id"] for d in docs])
    return reciprocal_rank_fusion(ranked_lists)[:k]
