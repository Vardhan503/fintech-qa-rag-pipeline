"""Shared LangChain models: HuggingFace embeddings + local Ollama LLM."""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "qwen2.5:7b-instruct"


def load_embedder(model_name: str = EMBED_MODEL) -> HuggingFaceEmbeddings:
    """CPU-forced so Apple Silicon MPS does not crash mid-encode."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )


def load_llm(model_name: str = LLM_MODEL) -> ChatOllama:
    return ChatOllama(model=model_name, temperature=0.0, num_predict=300)
