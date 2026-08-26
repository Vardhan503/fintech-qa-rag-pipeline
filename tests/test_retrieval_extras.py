"""Unit tests for shared fusion / NDCG helpers."""

from src.hybrid_search.ndcg import ndcg_at_k
from src.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_prefers_docs_that_rank_high_in_multiple_lists():
    a = ["x", "y", "z"]
    b = ["y", "x", "w"]
    fused = reciprocal_rank_fusion([a, b])
    assert fused[0] in ("x", "y")
    assert "w" in fused


def test_ndcg_perfect_ranking_is_one():
    gold = {"a", "b"}
    assert ndcg_at_k(["a", "b", "c"], gold, k=2) == 1.0


def test_ndcg_miss_is_zero():
    assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0
