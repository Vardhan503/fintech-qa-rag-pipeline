"""Query Rewriting for FinQA -- bonus technique.

Honest framing: FinQA's 883 questions are each standalone (no multi-turn
structure), so there's no follow-up-question failure mode to measure in the
benchmark itself. This still matters for FinQA-based work, though -- the
moment you put this behind a chat UI instead of a single-shot eval script,
real users WILL ask follow-ups ("what about last year?", "how does that
compare to..."). This module + the small hand-built conversational sample
in the notebook exist to validate the mechanism now, against real financial
question phrasing, before you need it live.

Uses LangChain's own `create_history_aware_retriever` -- built for exactly
this: given chat history + a new input, it decides whether the input needs
reformulating (skips the LLM call entirely if there's no history, or if the
input already looks standalone) and retrieves with the reformulated query.
"""
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given a chat history and the latest user question about a company's "
     "financial report, which might reference context in the chat history, "
     "reformulate it into a standalone question that can be understood "
     "without the chat history. Resolve pronouns and implicit references "
     "(e.g. 'that', 'it', 'the other year') explicitly. Do NOT answer the "
     "question, just reformulate it if needed, otherwise return it as-is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def build_history_aware_retriever(vectorstore: VectorStore, llm: BaseLanguageModel, k: int = 5):
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return create_history_aware_retriever(llm, base_retriever, CONTEXTUALIZE_PROMPT)


def rewrite_search(history_aware_retriever, question: str, chat_history: list) -> list[str]:
    docs = history_aware_retriever.invoke({"input": question, "chat_history": chat_history})
    return [d.metadata["chunk_id"] for d in docs]
