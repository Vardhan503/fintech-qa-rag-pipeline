"""NDCG@k with binary relevance (gold chunk ids)."""

import math


def ndcg_at_k(retrieved_ids, gold_ids, k: int) -> float:
    top_k = retrieved_ids[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(top_k, start=1)
        if doc_id in gold_ids
    )
    ideal = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0
