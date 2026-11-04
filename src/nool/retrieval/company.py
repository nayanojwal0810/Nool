"""Search current company passages with live document access checks."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from nool.data.documents import DocumentStore
from nool.retrieval.bm25 import BM25Retriever
from nool.retrieval.contriever import ContrieverRetriever
from nool.retrieval.hybrid import reciprocal_rank_fusion

DEMO_USERS = {
    "employee_demo": "employee",
    "hr_demo": "hr_admin",
    "it_admin_demo": "it_admin",
}


@dataclass(frozen=True)
class CompanySearchResult:
    document_id: str
    title: str
    chunk_id: str
    section: str
    page_number: int | None
    version: int
    text: str
    score: float
    rank: int
    sources: tuple[str, ...]


def export_current_corpus(store: DocumentStore, path: Path | str) -> int:
    """Write one deterministic search record per active passage."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for document in store.list_documents():
            if not document.active:
                continue
            for chunk in store.current_chunks(document.document_id):
                record = {
                    "_id": chunk.chunk_id,
                    "title": document.title,
                    "text": chunk.text,
                    "metadata": {
                        "document_id": document.document_id,
                        "version": chunk.version,
                        "content_hash": chunk.content_hash,
                        "section": chunk.section,
                        "page_number": chunk.page_number,
                    },
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    return count


def corpus_fingerprint(store: DocumentStore) -> str:
    """Identify the current searchable content without including role changes."""
    state = [
        (doc.document_id, doc.version, doc.content_hash, doc.active)
        for doc in store.list_documents()
    ]
    return hashlib.sha256(json.dumps(state).encode()).hexdigest()


class CompanyHybridSearch:
    """Fuse indexed passage results and verify every returned passage."""

    def __init__(
        self,
        store: DocumentStore,
        bm25: BM25Retriever,
        contriever: ContrieverRetriever,
        *,
        indexed_fingerprint: str,
        rank_constant: int = 20,
    ) -> None:
        self.store = store
        self.bm25 = bm25
        self.contriever = contriever
        self.indexed_fingerprint = indexed_fingerprint
        self.rank_constant = rank_constant

    def search(
        self, query: str, user_id: str, top_k: int = 10
    ) -> list[CompanySearchResult]:
        if user_id not in DEMO_USERS:
            raise ValueError("Unknown demo user")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if corpus_fingerprint(self.store) != self.indexed_fingerprint:
            raise RuntimeError("Company search index is outdated; rebuild it")

        role = DEMO_USERS[user_id]
        documents = {doc.document_id: doc for doc in self.store.list_documents()}
        chunks = {
            chunk.chunk_id: chunk
            for doc in documents.values()
            if doc.active and role in doc.allowed_roles
            for chunk in self.store.current_chunks(doc.document_id)
        }
        depth = max(len(self.bm25.documents), len(self.contriever.document_ids))

        def approved(results: list) -> list:
            safe = []
            for result in results:
                chunk = chunks.get(result.document_id)
                if chunk is None:
                    continue
                document = documents[chunk.document_id]
                if (
                    chunk.version == document.version
                    and chunk.content_hash == document.content_hash
                    and result.text == chunk.text
                    and result.title == document.title
                ):
                    safe.append(result)
            return safe

        fused = reciprocal_rank_fusion(
            approved(self.bm25.search(query, top_k=depth)),
            approved(self.contriever.search(query, top_k=depth)),
            rank_constant=self.rank_constant,
            top_k=depth,
        )
        results = []
        seen_documents: set[str] = set()
        for item in fused:
            chunk = chunks[item.document_id]
            if chunk.document_id in seen_documents:
                continue
            seen_documents.add(chunk.document_id)
            results.append(
                CompanySearchResult(
                    document_id=chunk.document_id,
                    title=item.title,
                    chunk_id=item.document_id,
                    section=chunk.section,
                    page_number=chunk.page_number,
                    version=chunk.version,
                    text=item.text,
                    score=item.score,
                    rank=len(results) + 1,
                    sources=item.sources,
                )
            )
            if len(results) == top_k:
                break
        return results
