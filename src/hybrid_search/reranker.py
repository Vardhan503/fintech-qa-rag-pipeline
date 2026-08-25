"""Cross-encoder reranking for FinQA -- the real thing this time (the
sandbox demo used a rarity-weighted overlap stand-in; this is
sentence-transformers' actual CrossEncoder, which needs HF model download
and doesn't run in this sandbox, but will on your machine since Project 4's
real embeddings already proved HF access works there).

`cross-encoder/ms-marco-MiniLM-L-6-v2` is a general-purpose, well-validated,
lightweight reranker (fast on CPU, ~80MB) -- a reasonable default. If you
want to try a larger/more accurate one: `BAAI/bge-reranker-v2-m3` (larger,
slower, no finance-specific reranker exists off the shelf, so general-purpose
is the honest choice here rather than reaching for something branded
'financial' that isn't actually validated for it).
"""
from sentence_transformers import CrossEncoder


def build_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    return CrossEncoder(model_name)


def rerank(query: str, candidates: list[dict], model: CrossEncoder, top_k: int = 5) -> list[str]:
    """candidates: list of {"chunk_id": ..., "text": ...} dicts (already-retrieved
    top-N from the hybrid stage, NOT the whole corpus -- that's the point of
    the retrieve-cheap-then-rerank-expensive pattern)."""
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c["chunk_id"] for _, c in scored[:top_k]]
