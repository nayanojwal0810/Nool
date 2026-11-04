"""Company passage search and access checks."""

import json
from pathlib import Path

import pytest

from nool.data.documents import DocumentStore
from nool.retrieval.bm25 import BM25SearchResult
from nool.retrieval.company import (
    CompanyHybridSearch,
    corpus_fingerprint,
    export_current_corpus,
)


class RankedStub:
    def __init__(self, results: list[BM25SearchResult]) -> None:
        self.results = results
        self.documents = results
        self.document_ids = [item.document_id for item in results]

    def search(self, query: str, top_k: int = 10) -> list[BM25SearchResult]:
        return self.results[:top_k]


def test_company_search_blocks_restricted_and_stale_passages(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path / "documents.sqlite")
    sources = []
    for doc_id, role, text in (
        ("OPEN", "employee", "Annual leave is 20 days."),
        ("PRIVATE", "hr_admin", "The private payroll code is red."),
    ):
        path = tmp_path / f"{doc_id}.md"
        path.write_text(f"# {doc_id}\n\n## Policy\n\n{text}", encoding="utf-8")
        store.ingest(
            path,
            document_id=doc_id,
            title=doc_id,
            owner="People",
            allowed_roles=[role],
        )
        sources.append(path)

    corpus = tmp_path / "corpus.jsonl"
    assert export_current_corpus(store, corpus) == 4
    records = [json.loads(line) for line in corpus.read_text().splitlines()]
    assert all("content_hash" in row["metadata"] for row in records)
    ranked = [
        BM25SearchResult(row["_id"], row["title"], row["text"], 1.0, rank)
        for rank, row in enumerate(reversed(records), start=1)
    ]
    stub = RankedStub(ranked)
    search = CompanyHybridSearch(
        store, stub, stub, indexed_fingerprint=corpus_fingerprint(store)
    )
    employee = search.search("leave", "employee_demo")
    assert employee
    assert {item.document_id for item in employee} == {"OPEN"}
    assert len(employee) == 1
    assert employee[0].chunk_id in {
        row["_id"] for row in records if row["metadata"]["document_id"] == "OPEN"
    }
    assert {item.document_id for item in search.search("payroll", "hr_demo")} == {
        "PRIVATE"
    }
    with pytest.raises(ValueError, match="Unknown demo user"):
        search.search("leave", "unknown")

    store.set_allowed_roles("OPEN", ["hr_admin"])
    assert search.search("leave", "employee_demo") == []
    assert {item.document_id for item in search.search("leave", "hr_demo")} == {
        "OPEN",
        "PRIVATE",
    }

    sources[0].write_text("# OPEN\n\n## Policy\n\nAnnual leave is 22 days.")
    store.ingest(
        sources[0],
        document_id="OPEN",
        title="OPEN",
        owner="People",
        allowed_roles=["hr_admin"],
    )
    with pytest.raises(RuntimeError, match="outdated"):
        search.search("leave", "hr_demo")
