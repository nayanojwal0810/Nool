from dataclasses import dataclass

import pytest

from nool.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


@dataclass
class Result:
    document_id: str
    title: str
    text: str


def test_shared_passage_ranks_above_single_source_passages():
    bm25 = [Result("a", "A", "alpha"), Result("b", "B", "beta")]
    dense = [Result("b", "B", "beta"), Result("c", "C", "gamma")]
    fused = reciprocal_rank_fusion(bm25, dense, rank_constant=60)
    assert [item.document_id for item in fused] == ["b", "a", "c"]
    assert fused[0].sources == ("bm25", "contriever")
    assert [item.rank for item in fused] == [1, 2, 3]


def test_empty_results_and_deterministic_tie():
    assert reciprocal_rank_fusion([], []) == []
    fused = reciprocal_rank_fusion(
        [Result("b", "B", "body")], [Result("a", "A", "body")]
    )
    assert [item.document_id for item in fused] == ["a", "b"]


def test_conflicting_content_fails():
    with pytest.raises(ValueError, match="Conflicting"):
        reciprocal_rank_fusion([Result("a", "A", "one")], [Result("a", "A", "two")])


def test_hybrid_calls_both_retrievers():
    class Retriever:
        def __init__(self, results):
            self.results = results
            self.calls = []

        def search(self, query, top_k):
            self.calls.append((query, top_k))
            return self.results

    bm25 = Retriever([Result("a", "A", "alpha")])
    dense = Retriever([Result("a", "A", "alpha")])
    hybrid = HybridRetriever(bm25, dense, candidate_k=10)
    assert hybrid.search("question", top_k=3)[0].document_id == "a"
    assert bm25.calls == dense.calls == [("question", 10)]
