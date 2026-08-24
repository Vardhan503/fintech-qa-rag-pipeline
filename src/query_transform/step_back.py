"""Step-Back Prompting for FinQA.

Unlike HyDE and multi-query, LangChain doesn't ship an official class for
this -- built with a standard LCEL chain (ChatPromptTemplate | llm |
StrOutputParser), which is still 100% library machinery; only the prompt
text itself is hand-authored, same as your roadmap's own tech-stack note
("custom HyDE/step-back prompts").

Few-shot examples are in-domain (financial), which matters -- generic
few-shot examples produce step-back questions that are too generic to be
useful against a financial-report corpus.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore

from src.query_transform.fusion import reciprocal_rank_fusion

STEP_BACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You generate a more general 'step-back' version of a specific financial "
     "question, one level more abstract, still specific enough to be useful "
     "for searching a financial report's tables and text.\n\n"
     "Example 1:\n"
     "Original: what is the average payment volume per transaction for american express?\n"
     "Step-back: what are the payment volume and transaction figures reported for american express?\n\n"
     "Example 2:\n"
     "Original: what percentage of the total oil and gas mmboe comes from canada?\n"
     "Step-back: what are the oil and gas mmboe figures reported by country/region?\n\n"
     "Return ONLY the step-back question, nothing else."),
    ("human", "{question}"),
])


def build_step_back_chain(llm: BaseLanguageModel):
    return STEP_BACK_PROMPT | llm | StrOutputParser()


def step_back_search(vectorstore: VectorStore, question: str, step_back_chain, k: int = 5) -> list[str]:
    general_question = step_back_chain.invoke({"question": question}).strip()
    original_docs = vectorstore.similarity_search(question, k=k)
    general_docs = vectorstore.similarity_search(general_question, k=k)
    original_ids = [d.metadata["chunk_id"] for d in original_docs]
    general_ids = [d.metadata["chunk_id"] for d in general_docs]
    return reciprocal_rank_fusion([original_ids, general_ids])[:k]
