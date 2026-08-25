"""Reciprocal Rank Fusion -- merges 2+ ranked chunk_id lists into one, by
rank position rather than raw similarity score (scores from different
retrieval passes aren't on comparable scales). k=60 is the standard
constant from Cormack et al. 2009; it dampens the score gap between rank 1
and rank 2 so no single list dominates the fused ranking.

Reused as-is for Project 5 (BM25 + dense hybrid fusion).
"""
from collections import defaultdict


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
