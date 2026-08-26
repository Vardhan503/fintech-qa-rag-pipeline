"""Reciprocal Rank Fusion (Cormack et al., 2009).

Merges ranked id lists by position, not raw score, so BM25 and dense
results can be combined without score calibration. k=60 is the usual constant.
"""
from collections import defaultdict


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])]
