"""History-aware rewrite for follow-up questions (chat / multi-turn use)."""

from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import VectorStore

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given chat history and the latest user question about a financial "
        "report, rewrite it as a standalone question. Resolve pronouns and "
        "references like 'that' or 'last year'. Do not answer — only rewrite "
        "if needed, otherwise return the question as-is.",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def build_history_aware_retriever(vectorstore: VectorStore, llm: BaseLanguageModel, k: int = 5):
    base = vectorstore.as_retriever(search_kwargs={"k": k})
    return create_history_aware_retriever(llm, base, CONTEXTUALIZE_PROMPT)


def rewrite_search(history_aware_retriever, question: str, chat_history: list) -> list[str]:
    docs = history_aware_retriever.invoke({"input": question, "chat_history": chat_history})
    return [d.metadata["chunk_id"] for d in docs]
