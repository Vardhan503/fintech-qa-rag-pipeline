"""Cross-encoder reranker over a short candidate list."""

from sentence_transformers import CrossEncoder


def build_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    return CrossEncoder(model_name)


def rerank(query: str, candidates: list[dict], model: CrossEncoder, top_k: int = 5) -> list[str]:
    """candidates: [{chunk_id, text}, ...] from a cheap first-stage retrieve."""
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    scored = sorted(zip(scores, candidates), key=lambda x: -x[0])
    return [c["chunk_id"] for _, c in scored[:top_k]]
