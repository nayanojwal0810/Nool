"""Document changes and access changes across the company search boundary."""

import json
from pathlib import Path

import pytest
import yaml
from pypdf import PdfReader, PdfWriter

from nool.config import get_project_root
from nool.data.collection import sync_collection
from nool.data.documents import DocumentStore
from nool.retrieval.bm25 import BM25SearchResult
from nool.retrieval.company import (
    CompanyHybridSearch,
    corpus_fingerprint,
    export_current_corpus,
    index_matches_store,
)


class RankedStub:
    def __init__(self, rows: list[dict]) -> None:
        self.results = [
            BM25SearchResult(row["_id"], row["title"], row["text"], 1.0, rank)
            for rank, row in enumerate(rows, start=1)
        ]
        self.documents = self.results
        self.document_ids = [item.document_id for item in self.results]

    def search(self, query: str, top_k: int = 10) -> list[BM25SearchResult]:
        return self.results[:top_k]


def write_manifest(root: Path, entries: list[dict]) -> Path:
    path = root / "manifest.yaml"
    path.write_text(
        yaml.safe_dump({"company": "Example", "documents": entries}), encoding="utf-8"
    )
    return path


def entry(document_id: str, suffix: str, role: str = "employee") -> dict:
    return {
        "document_id": document_id,
        "title": document_id,
        "source": f"{document_id}{suffix}",
        "owner": "People",
        "allowed_roles": [role],
        "version": 1,
        "effective_date": "2026-01-01",
    }


def indexed_search(store: DocumentStore, corpus: Path) -> CompanyHybridSearch:
    export_current_corpus(store, corpus)
    rows = [
        json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()
    ]
    stub = RankedStub(rows)
    return CompanyHybridSearch(
        store, stub, stub, indexed_fingerprint=corpus_fingerprint(store)
    )


def test_access_changes_apply_without_rebuilding(tmp_path: Path) -> None:
    public = tmp_path / "OPEN.md"
    public.write_text("# OPEN\n\nThe office opens at nine.", encoding="utf-8")
    private = tmp_path / "PRIVATE.txt"
    private.write_text("PRIVATE\n\nPrivate payroll review.", encoding="utf-8")
    entries = [entry("OPEN", ".md"), entry("PRIVATE", ".txt", "hr_admin")]
    manifest = write_manifest(tmp_path, entries)
    database = tmp_path / "documents.sqlite"
    first = sync_collection(manifest, database)
    assert first.created == ("OPEN", "PRIVATE")
    assert first.requires_rebuild
    store = DocumentStore(database)
    search = indexed_search(store, tmp_path / "corpus.jsonl")
    assert {item.document_id for item in search.search("policy", "employee_demo")} == {
        "OPEN"
    }
    assert {item.document_id for item in search.search("policy", "hr_demo")} == {
        "PRIVATE"
    }
    with pytest.raises(ValueError, match="Unknown demo user"):
        search.search("policy", "stranger")

    assert not sync_collection(manifest, database).requires_rebuild
    entries[0]["allowed_roles"] = ["hr_admin"]
    changed = sync_collection(write_manifest(tmp_path, entries), database)
    assert changed.access_changed == ("OPEN",)
    assert not changed.requires_rebuild
    assert search.search("policy", "employee_demo") == []
    assert {item.document_id for item in search.search("policy", "hr_demo")} == {
        "OPEN",
        "PRIVATE",
    }
    entries[0]["allowed_roles"] = ["employee"]
    assert not sync_collection(
        write_manifest(tmp_path, entries), database
    ).requires_rebuild
    assert {item.document_id for item in search.search("policy", "employee_demo")} == {
        "OPEN"
    }


@pytest.mark.parametrize("suffix", [".md", ".txt", ".pdf"])
def test_content_update_blocks_old_index_until_rebuild(
    tmp_path: Path, suffix: str
) -> None:
    if suffix == ".pdf":
        original = (
            get_project_root() / "data/demo/documents/people/annual_leave_policy.pdf"
        )
        title = "Annual Leave Policy"
        document_id = "PEO-001"
        source = tmp_path / f"{document_id}{suffix}"
        source.write_bytes(original.read_bytes())
    else:
        document_id = "DOC"
        title = "DOC"
        source = tmp_path / f"{document_id}{suffix}"
        source.write_text("DOC\n\nAnnual leave is 20 days.", encoding="utf-8")
    record = entry(document_id, suffix)
    record["title"] = title
    manifest = write_manifest(tmp_path, [record])
    database = tmp_path / "documents.sqlite"
    sync_collection(manifest, database)
    store = DocumentStore(database)
    old_search = indexed_search(store, tmp_path / "old.jsonl")
    old_ids = {item.chunk_id for item in store.current_chunks(document_id)}

    if suffix == ".pdf":
        writer = PdfWriter()
        writer.append(PdfReader(source))
        writer.add_metadata({"/Revision": "2"})
        with source.open("wb") as stream:
            writer.write(stream)
    else:
        source.write_text("DOC\n\nAnnual leave is 22 days.", encoding="utf-8")
    changed = sync_collection(manifest, database)
    assert changed.content_updated == (document_id,)
    assert changed.requires_rebuild
    assert store.get_document(document_id).version == 2
    with pytest.raises(RuntimeError, match="outdated"):
        old_search.search("annual leave", "employee_demo")

    new_search = indexed_search(store, tmp_path / "new.jsonl")
    fresh = new_search.search("annual leave", "employee_demo")
    assert fresh
    assert not old_ids.intersection({item.chunk_id for item in fresh})
    if suffix != ".pdf":
        assert "22 days" in " ".join(item.text for item in fresh)


def test_archive_and_reactivate_require_rebuild(tmp_path: Path) -> None:
    for document_id in ("OPEN", "SECOND"):
        (tmp_path / f"{document_id}.md").write_text(
            f"# {document_id}\n\nCompany rule for {document_id}.", encoding="utf-8"
        )
    entries = [entry("OPEN", ".md"), entry("SECOND", ".md")]
    manifest = write_manifest(tmp_path, entries)
    database = tmp_path / "documents.sqlite"
    sync_collection(manifest, database)
    store = DocumentStore(database)
    old_search = indexed_search(store, tmp_path / "old.jsonl")

    removed = sync_collection(write_manifest(tmp_path, entries[:1]), database)
    assert removed.archived == ("SECOND",)
    assert removed.requires_rebuild
    assert store.current_chunks("SECOND") == []
    with pytest.raises(RuntimeError, match="outdated"):
        old_search.search("rule", "employee_demo")

    restored = sync_collection(write_manifest(tmp_path, entries), database)
    assert restored.reactivated == ("SECOND",)
    assert restored.requires_rebuild
    assert store.get_document("SECOND").version == 1
    assert {
        item.document_id
        for item in indexed_search(store, tmp_path / "new.jsonl").search(
            "rule", "employee_demo"
        )
    } == {"OPEN", "SECOND"}


def test_title_change_blocks_old_search(tmp_path: Path) -> None:
    source = tmp_path / "DOC.md"
    source.write_text("# Updated DOC\n\nCompany rule.", encoding="utf-8")
    record = entry("DOC", ".md")
    manifest = write_manifest(tmp_path, [record])
    database = tmp_path / "documents.sqlite"
    sync_collection(manifest, database)
    store = DocumentStore(database)
    old_search = indexed_search(store, tmp_path / "old.jsonl")

    record["title"] = "Updated DOC"
    changed = sync_collection(write_manifest(tmp_path, [record]), database)
    assert changed.title_updated == ("DOC",)
    assert changed.requires_rebuild
    with pytest.raises(RuntimeError, match="outdated"):
        old_search.search("rule", "employee_demo")


def test_invalid_manifest_leaves_access_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "DOC.md"
    source.write_text("# DOC\n\nCompany rule.", encoding="utf-8")
    record = entry("DOC", ".md")
    manifest = write_manifest(tmp_path, [record])
    database = tmp_path / "documents.sqlite"
    sync_collection(manifest, database)
    record["allowed_roles"] = ["unknown"]
    with pytest.raises(ValueError, match="Invalid allowed roles"):
        sync_collection(write_manifest(tmp_path, [record]), database)
    assert DocumentStore(database).get_document("DOC").allowed_roles == ("employee",)


def test_content_and_access_change_together(tmp_path: Path) -> None:
    source = tmp_path / "DOC.txt"
    source.write_text("DOC\n\nOld instruction.", encoding="utf-8")
    record = entry("DOC", ".txt")
    manifest = write_manifest(tmp_path, [record])
    database = tmp_path / "documents.sqlite"
    sync_collection(manifest, database)
    store = DocumentStore(database)
    old_search = indexed_search(store, tmp_path / "old.jsonl")

    source.write_text("DOC\n\nNew instruction.", encoding="utf-8")
    record["allowed_roles"] = ["hr_admin"]
    changed = sync_collection(write_manifest(tmp_path, [record]), database)
    assert changed.content_updated == ("DOC",)
    assert changed.access_changed == ("DOC",)
    with pytest.raises(RuntimeError, match="outdated"):
        old_search.search("instruction", "employee_demo")
    fresh = indexed_search(store, tmp_path / "new.jsonl")
    assert fresh.search("instruction", "employee_demo") == []
    assert "New instruction" in fresh.search("instruction", "hr_demo")[0].text


def test_index_status_reflects_prior_changes(tmp_path: Path) -> None:
    source = tmp_path / "DOC.md"
    source.write_text("# Updated DOC\n\nCompany rule.", encoding="utf-8")
    record = entry("DOC", ".md")
    manifest = write_manifest(tmp_path, [record])
    database = tmp_path / "documents.sqlite"
    sync_collection(manifest, database)
    store = DocumentStore(database)
    index = tmp_path / "index"
    export_current_corpus(store, index / "corpus.jsonl")
    (index / "source_fingerprint.txt").write_text(
        corpus_fingerprint(store), encoding="utf-8"
    )
    assert index_matches_store(store, index)

    record["allowed_roles"] = ["hr_admin"]
    sync_collection(write_manifest(tmp_path, [record]), database)
    assert index_matches_store(store, index)

    record["title"] = "Updated DOC"
    sync_collection(write_manifest(tmp_path, [record]), database)
    assert not index_matches_store(store, index)
    assert not sync_collection(manifest, database).requires_rebuild
    assert not index_matches_store(store, index)
