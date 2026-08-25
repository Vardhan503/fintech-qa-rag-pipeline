"""
NDCG@k, added for Project 5 (your roadmap's benchmark metric here).

With binary relevance (a doc is either gold or not -- which is what our
eval sets use), NDCG@k reduces to a clean, interpretable formula:

    DCG@k  = sum over relevant docs in top-k of  1 / log2(rank + 1)
    IDCG@k = the DCG@k you'd get if ALL gold docs were ranked first
    NDCG@k = DCG@k / IDCG@k

It's like Recall@k but rank-SENSITIVE: a relevant doc at rank 1 counts more
than the same doc at rank 5. Recall@k treats "found it at rank 1" and
"found it at rank 5" identically as long as k is large enough; NDCG doesn't.
"""
import math

def ndcg_at_k(retrieved_ids, gold_ids, k):
    top_k = retrieved_ids[:k]
    dcg = sum(1.0 / math.log2(rank + 1) for rank, d in enumerate(top_k, start=1) if d in gold_ids)
    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
