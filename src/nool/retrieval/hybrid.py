"""Hybrid retrieval using reciprocal rank fusion of BM25 and Contriever."""

from dataclasses import dataclass

from nool.retrieval.bm25 import BM25Retriever
from nool.retrieval.contriever import ContrieverRetriever


@dataclass(frozen=True)
class HybridSearchResult:
    document_id: str
    title: str
    text: str
    score: float
    rank: int
    sources: tuple[str, ...]


def reciprocal_rank_fusion(
    bm25_results: list,
    contriever_results: list,
    *,
    rank_constant: int = 60,
    top_k: int = 10,
) -> list[HybridSearchResult]:
    """Fuse two ranked lists, deduplicate IDs, and break ties deterministically."""
    if rank_constant <= 0 or isinstance(rank_constant, bool):
        raise ValueError("rank_constant must be a positive integer")
    if top_k <= 0 or isinstance(top_k, bool):
        raise ValueError("top_k must be a positive integer")

    scores: dict[str, float] = {}
    passages: dict[str, tuple[str, str]] = {}
    sources: dict[str, list[str]] = {}
    first_rank: dict[str, int] = {}
    for source_name, results in (
        ("bm25", bm25_results),
        ("contriever", contriever_results),
    ):
        seen: set[str] = set()
        for position, result in enumerate(results, start=1):
            document_id = result.document_id
            if document_id in seen:
                raise ValueError(
                    f"Duplicate document ID in {source_name} results: {document_id}"
                )
            seen.add(document_id)
            passage = (result.title, result.text)
            if document_id in passages and passages[document_id] != passage:
                raise ValueError(
                    f"Conflicting passage content for document ID {document_id}"
                )
            passages[document_id] = passage
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (
                rank_constant + position
            )
            sources.setdefault(document_id, []).append(source_name)
            first_rank[document_id] = min(
                first_rank.get(document_id, position), position
            )

    ordered_ids = sorted(
        scores, key=lambda doc_id: (-scores[doc_id], first_rank[doc_id], doc_id)
    )
    return [
        HybridSearchResult(
            document_id=doc_id,
            title=passages[doc_id][0],
            text=passages[doc_id][1],
            score=scores[doc_id],
            rank=rank,
            sources=tuple(sources[doc_id]),
        )
        for rank, doc_id in enumerate(ordered_ids[:top_k], start=1)
    ]


class HybridRetriever:
    """Run both retrievers and return one fused ranking."""

    def __init__(
        self,
        bm25: BM25Retriever,
        contriever: ContrieverRetriever,
        *,
        rank_constant: int = 60,
        candidate_k: int = 10,
    ) -> None:
        if candidate_k <= 0 or isinstance(candidate_k, bool):
            raise ValueError("candidate_k must be a positive integer")
        self.bm25 = bm25
        self.contriever = contriever
        self.rank_constant = rank_constant
        self.candidate_k = candidate_k

    def search(self, query: str, top_k: int = 10) -> list[HybridSearchResult]:
        depth = max(self.candidate_k, top_k)
        return reciprocal_rank_fusion(
            self.bm25.search(query, top_k=depth),
            self.contriever.search(query, top_k=depth),
            rank_constant=self.rank_constant,
            top_k=top_k,
        )
