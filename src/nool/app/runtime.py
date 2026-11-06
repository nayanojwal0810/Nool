"""Load the company answer service from the published demo collection."""

import hashlib
import zipfile
from pathlib import Path

from nool.data.collection import ingest_collection, sync_collection
from nool.data.documents import DocumentStore
from nool.generation.answers import CompanyAnswerService
from nool.generation.ollama import OllamaAnswerModel
from nool.retrieval.bm25 import BM25Retriever
from nool.retrieval.company import CompanyHybridSearch, index_matches_store
from nool.retrieval.contriever import (
    ContrieverEncoder,
    ContrieverRetriever,
    compute_sha256,
)


def _extract_index(archive_path: Path, destination: Path) -> Path:
    index_path = destination / "company"
    if (index_path / "source_fingerprint.txt").is_file():
        return index_path
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Search archive contains an unsafe path")
        archive.extractall(destination)
    return index_path


def load_answer_service(
    project_root: Path,
    runtime_root: Path,
    *,
    model: str = "qwen3:4b-instruct",
    model_url: str = "http://127.0.0.1:11434",
) -> CompanyAnswerService:
    """Build the runtime database and open the matching company index."""
    project_root = project_root.resolve()
    archive_path = project_root / "indexes/company_search_artifact.zip"
    manifest_path = project_root / "data/demo/manifest.yaml"
    digest = hashlib.sha256(
        archive_path.read_bytes() + manifest_path.read_bytes()
    ).hexdigest()[:16]
    runtime_path = runtime_root.resolve() / digest
    index_path = _extract_index(archive_path, runtime_path / "indexes")
    database_path = runtime_path / "demo_documents.sqlite"
    if not database_path.is_file():
        ingest_collection(manifest_path, database_path)
    else:
        sync_collection(manifest_path, database_path)
    store = DocumentStore(database_path)
    if not index_matches_store(store, index_path):
        raise RuntimeError("Company documents and search index do not match")
    corpus_path = index_path / "corpus.jsonl"
    bm25 = BM25Retriever.load(index_path / "bm25")
    if bm25.manifest["source_corpus_sha256"] != compute_sha256(corpus_path):
        raise RuntimeError("Company search indexes do not match")
    dense = ContrieverRetriever.load(
        index_path / "contriever",
        corpus_path,
        encoder=ContrieverEncoder(),
    )
    fingerprint = (index_path / "source_fingerprint.txt").read_text(encoding="utf-8")
    search = CompanyHybridSearch(store, bm25, dense, indexed_fingerprint=fingerprint)
    return CompanyAnswerService(
        search,
        store,
        OllamaAnswerModel(model=model, base_url=model_url),
    )
