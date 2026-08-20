"""Retrieval-only metrics: embed, cosine search, precision/recall @ k.

Two evaluators live here and they are not interchangeable:

  evaluate_chunking_strategy         exact chunk_id match against gold ids. Only
                                    meaningful for strategies that reuse the
                                    row-level id scheme, i.e. the baseline.
  evaluate_chunking_strategy_generic content-overlap match. Valid ACROSS
                                    strategies with different granularities and
                                    id schemes -- use this for any cross-strategy
                                    comparison, since a whole_table chunk can
                                    never equal a table_row gold id by id alone.
"""

import numpy as np
from langchain_core.embeddings import Embeddings

from src.models import EMBED_MODEL, load_embedder  # noqa: F401 — re-exported
from src.utils.io import load_jsonl

MODEL_NAME = EMBED_MODEL


def load_index_chunks(path) -> list[dict]:
    """Chunks as they enter the index: noise lines are never retrievable."""
    return [c for c in load_jsonl(path) if not c.get("is_noise", False)]


def embed_texts(texts: list[str], model: Embeddings, batch_size: int = 64,
                show_progress_bar: bool = True) -> np.ndarray:
    """Returns L2-normalized embeddings so dot product == cosine similarity."""
    del batch_size, show_progress_bar  # HuggingFaceEmbeddings encode_kwargs already set these
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(model.embed_documents(texts), dtype=np.float32)


def cosine_top_k(query_vec, chunk_vecs, chunk_ids, k: int = 5) -> list[str]:
    sims = chunk_vecs @ query_vec
    top_k_idx = np.argsort(-sims)[:k]
    return [chunk_ids[i] for i in top_k_idx]


def context_precision_at_k(retrieved_ids, gold_ids, k: int) -> float:
    top_k = retrieved_ids[:k]
    return sum(1 for c in top_k if c in gold_ids) / k if top_k else 0.0


def context_recall_at_k(retrieved_ids, gold_ids, k: int):
    if not gold_ids:
        return None
    top_k = set(retrieved_ids[:k])
    return sum(1 for g in gold_ids if g in top_k) / len(gold_ids)


def word_overlap_relevance(retrieved_text: str, gold_text: str, threshold: float = 0.6) -> bool:
    gold_words = set(w.lower().strip(".,;:()") for w in gold_text.split())
    retrieved_words = set(w.lower().strip(".,;:()") for w in retrieved_text.split())
    if not gold_words:
        return False
    overlap = len(gold_words & retrieved_words) / len(gold_words)
    return overlap >= threshold


def evaluate_chunking_strategy(chunks: list[dict], eval_examples: list[dict],
                                model: Embeddings, k_values=(1, 3, 5, 10)) -> dict:
    """Exact-id scoring. chunks: [{chunk_id, text, ...}], eval_examples: [{question, gold_chunk_ids}]."""
    chunk_ids = [c["chunk_id"] for c in chunks]
    chunk_texts = [c["text"] for c in chunks]
    chunk_vecs = embed_texts(chunk_texts, model)

    query_vecs = embed_texts([e["question"] for e in eval_examples], model)

    results = {k: {"precision": [], "recall": []} for k in k_values}
    max_k = max(k_values)

    for i, ex in enumerate(eval_examples):
        gold_ids = set(ex["gold_chunk_ids"])
        retrieved = cosine_top_k(query_vecs[i], chunk_vecs, chunk_ids, k=max_k)
        for k in k_values:
            results[k]["precision"].append(context_precision_at_k(retrieved, gold_ids, k))
            r = context_recall_at_k(retrieved, gold_ids, k)
            if r is not None:
                results[k]["recall"].append(r)

    return {k: {"precision": float(np.mean(v["precision"])), "recall": float(np.mean(v["recall"]))}
            for k, v in results.items()}


def gold_text_index(row_level_chunks: list[dict]) -> dict[str, str]:
    """gold chunk_id -> its original row-level text, the bridge that lets a
    differently-chunked index be scored against row-level gold references."""
    return {c["chunk_id"]: c["text"] for c in row_level_chunks}


def recall_at_k_by_content(chunks: list[dict], eval_examples: list[dict],
                            chunk_id_to_gold_text: dict, model: Embeddings,
                            k: int = 5) -> dict:
    """Single-k content-overlap recall, for sweeping a chunking parameter.

    Questions with no gold annotation (65 of the 883 dev questions) are excluded
    from the denominator rather than counted as misses, which is also how
    evaluate_chunking_strategy_generic averages -- keeping the two comparable.
    """
    chunk_ids = [c["chunk_id"] for c in chunks]
    chunk_texts = [c["text"] for c in chunks]
    chunk_vecs = embed_texts(chunk_texts, model)
    query_vecs = embed_texts([e["question"] for e in eval_examples], model)
    id_to_text = dict(zip(chunk_ids, chunk_texts))

    found, scored = 0, 0
    for i, ex in enumerate(eval_examples):
        gold_texts = [chunk_id_to_gold_text[g] for g in ex["gold_chunk_ids"] if g in chunk_id_to_gold_text]
        if not gold_texts:
            continue
        scored += 1

        retrieved_ids = cosine_top_k(query_vecs[i], chunk_vecs, chunk_ids, k=k)
        if any(word_overlap_relevance(id_to_text[cid], gt)
               for cid in retrieved_ids for gt in gold_texts):
            found += 1

    return {
        "recall": found / scored if scored else 0.0,
        "n_scored": scored,
        "n_chunks": len(chunks),
    }


def evaluate_chunking_strategy_generic(chunks: list[dict], eval_examples: list[dict],
                                        chunk_id_to_gold_text: dict, model: Embeddings,
                                        k_values=(1, 3, 5, 10)) -> dict:
    """Content-overlap scoring, valid across chunking strategies."""
    chunk_ids = [c["chunk_id"] for c in chunks]
    chunk_texts = [c["text"] for c in chunks]
    id_to_text = dict(zip(chunk_ids, chunk_texts))
    chunk_vecs = embed_texts(chunk_texts, model)

    query_vecs = embed_texts([e["question"] for e in eval_examples], model)

    results = {k: {"precision": [], "recall": []} for k in k_values}
    max_k = max(k_values)

    for i, ex in enumerate(eval_examples):
        gold_texts = [chunk_id_to_gold_text[gid] for gid in ex["gold_chunk_ids"] if gid in chunk_id_to_gold_text]
        if not gold_texts:
            continue
        retrieved_ids = cosine_top_k(query_vecs[i], chunk_vecs, chunk_ids, k=max_k)

        for k in k_values:
            top_k_texts = [id_to_text[cid] for cid in retrieved_ids[:k]]
            # relevant if ANY retrieved chunk in top-k covers ANY gold fact
            relevant_flags = [
                any(word_overlap_relevance(rt, gt) for gt in gold_texts)
                for rt in top_k_texts
            ]
            results[k]["precision"].append(sum(relevant_flags) / k)
            # simplified: FinQA gold sets are near-always size 1, so coverage of
            # any gold fact within top-k counts as full recall
            results[k]["recall"].append(1.0 if any(relevant_flags) else 0.0)

    return {k: {"precision": float(np.mean(v["precision"])), "recall": float(np.mean(v["recall"]))}
            for k, v in results.items()}
