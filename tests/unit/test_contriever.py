"""Unit tests for Contriever dense retrieval, artifact loading, and verification."""

import json
from pathlib import Path

import numpy as np
import pytest

from nool.retrieval.bm25 import BM25Retriever
from nool.retrieval.contriever import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    ContrieverEncoder,
    ContrieverRetriever,
    DenseSearchResult,
    compute_sha256,
    mean_pooling,
    verify_artifact,
)


class FakeQueryEncoder:
    """Fake encoder that returns a predefined embedding vector without PyTorch."""

    def __init__(self, query_vector: np.ndarray) -> None:
        self.query_vector = query_vector
        self.recorded_query: str | None = None

    def encode_query(self, query: str) -> np.ndarray:
        self.recorded_query = query
        return self.query_vector


def _create_synthetic_artifact(
    tmp_path: Path,
    num_docs: int = 5,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    non_finite: bool = False,
    duplicate_ids: bool = False,
    missing_from_corpus: bool = False,
) -> tuple[Path, Path]:
    """Helper to generate a self-consistent synthetic Contriever artifact and corpus."""
    art_dir = tmp_path / "synthetic_artifact"
    art_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = tmp_path / "corpus.jsonl"

    # 1. Generate corpus records
    corpus_lines = []
    doc_ids = []
    for i in range(num_docs):
        doc_id = f"doc_{i}"
        doc_ids.append(doc_id)
        record = {
            "_id": doc_id,
            "title": f"Document Title {i}",
            "text": f"This is passage content for document {i}.",
            "metadata": {"doc_index": i},
        }
        corpus_lines.append(json.dumps(record))

    if missing_from_corpus and corpus_lines:
        # Exclude doc_0 from corpus file
        corpus_file.write_text("\n".join(corpus_lines[1:]) + "\n", encoding="utf-8")
    else:
        corpus_file.write_text("\n".join(corpus_lines) + "\n", encoding="utf-8")

    corpus_sha256 = compute_sha256(corpus_file)

    # 2. Generate embeddings
    # Make vectors deterministic: row i has i+1 in first position, rest 0
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((num_docs, embedding_dim)).astype(np.float32)
    if non_finite and num_docs > 0:
        embeddings[0, 0] = np.nan

    emb_file = art_dir / "passage_embeddings.npy"
    np.save(str(emb_file), embeddings)
    emb_sha256 = compute_sha256(emb_file)

    # 3. Save document IDs
    if duplicate_ids and len(doc_ids) > 1:
        saved_doc_ids = list(doc_ids)
        saved_doc_ids[1] = saved_doc_ids[0]
    else:
        saved_doc_ids = list(doc_ids)

    doc_ids_file = art_dir / "document_ids.json"
    doc_ids_file.write_text(json.dumps(saved_doc_ids, indent=2), encoding="utf-8")
    doc_ids_sha256 = compute_sha256(doc_ids_file)

    # 4. Save manifest
    manifest = {
        "artifact_schema_version": "1.0.0",
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": DEFAULT_MODEL_REVISION,
        "embedding_dimension": embedding_dim,
        "pooling": "masked_mean",
        "max_token_length": 512,
        "normalize_embeddings": False,
        "similarity_method": "dot_product",
        "output_dtype": "float32",
        "passage_count": num_docs,
        "source_corpus_filename": corpus_file.name,
        "source_corpus_sha256": corpus_sha256,
        "document_ids_file": "document_ids.json",
        "document_ids_sha256": doc_ids_sha256,
        "embedding_file": "passage_embeddings.npy",
        "embedding_sha256": emb_sha256,
        "pytorch_version": "2.15.0",
        "transformers_version": "5.22.1",
        "created_at": "2026-11-04T00:00:00Z",
    }
    manifest_file = art_dir / "index_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return art_dir, corpus_file


def test_valid_artifact_load_and_properties(tmp_path: Path) -> None:
    """Verify loading a valid artifact succeeds and populates properties."""
    art_dir, corpus_file = _create_synthetic_artifact(tmp_path, num_docs=5)

    retriever = ContrieverRetriever.load(art_dir, corpus_file)

    assert len(retriever.document_ids) == 5
    assert retriever.embeddings.shape == (5, DEFAULT_EMBEDDING_DIM)
    assert retriever.embeddings.dtype == np.float32
    assert retriever.manifest["model_id"] == DEFAULT_MODEL_ID
    assert retriever.manifest["model_revision"] == DEFAULT_MODEL_REVISION
    assert len(retriever.corpus_documents) == 5


def test_correct_dot_product_ranking_and_fields(tmp_path: Path) -> None:
    """Verify exact dot-product ranking, descending score order, and 1-based ranks."""
    art_dir, corpus_file = _create_synthetic_artifact(tmp_path, num_docs=4)

    # Set explicit orthogonal embeddings:
    # doc_0=[1, 0, ...], doc_1=[2, 0, ...], doc_2=[0.5, 0, ...], doc_3=[-1, 0, ...]
    emb = np.zeros((4, DEFAULT_EMBEDDING_DIM), dtype=np.float32)
    emb[0, 0] = 1.0
    emb[1, 0] = 2.0
    emb[2, 0] = 0.5
    emb[3, 0] = -1.0
    emb_file = art_dir / "passage_embeddings.npy"
    np.save(str(emb_file), emb)

    # Update embedding checksum in manifest
    manifest_file = art_dir / "index_manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["embedding_sha256"] = compute_sha256(emb_file)
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    query_vec = np.zeros(DEFAULT_EMBEDDING_DIM, dtype=np.float32)
    query_vec[0] = 2.0  # dot products: doc_1=4.0, doc_0=2.0, doc_2=1.0, doc_3=-2.0
    fake_encoder = FakeQueryEncoder(query_vec)

    retriever = ContrieverRetriever.load(art_dir, corpus_file, encoder=fake_encoder)
    results = retriever.search("sample query", top_k=4)

    assert len(results) == 4
    # Expected ranking: doc_1 (#1), doc_0 (#2), doc_2 (#3), doc_3 (#4)
    expected_ids = ["doc_1", "doc_0", "doc_2", "doc_3"]
    expected_scores = [4.0, 2.0, 1.0, -2.0]

    for idx, (res, exp_id, exp_score) in enumerate(
        zip(results, expected_ids, expected_scores, strict=True), start=1
    ):
        assert isinstance(res, DenseSearchResult)
        assert res.document_id == exp_id
        assert pytest.approx(res.score, 1e-5) == exp_score
        assert res.rank == idx
        assert res.title == f"Document Title {exp_id.split('_')[1]}"
        assert f"content for document {exp_id.split('_')[1]}" in res.text
        assert "doc_index" in res.metadata

    # Check descending order
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_and_oversized_handling(tmp_path: Path) -> None:
    """Verify top_k limits results, and top_k > doc_count returns all documents."""
    art_dir, corpus_file = _create_synthetic_artifact(tmp_path, num_docs=3)
    query_vec = np.zeros(DEFAULT_EMBEDDING_DIM, dtype=np.float32)
    fake_encoder = FakeQueryEncoder(query_vec)

    retriever = ContrieverRetriever.load(art_dir, corpus_file, encoder=fake_encoder)

    # top_k = 2 on 3 documents
    res_2 = retriever.search("query", top_k=2)
    assert len(res_2) == 2

    # top_k = 10 on 3 documents
    res_oversized = retriever.search("query", top_k=10)
    assert len(res_oversized) == 3


def test_blank_query_rejection(tmp_path: Path) -> None:
    """Verify empty, whitespace, or invalid queries are rejected."""
    art_dir, corpus_file = _create_synthetic_artifact(tmp_path, num_docs=2)
    retriever = ContrieverRetriever.load(
        art_dir, corpus_file, encoder=FakeQueryEncoder(np.zeros(DEFAULT_EMBEDDING_DIM))
    )

    with pytest.raises(ValueError, match="Query string cannot be empty"):
        retriever.search("")

    with pytest.raises(ValueError, match="Query string cannot be empty"):
        retriever.search("   ")

    with pytest.raises(ValueError, match="Query string cannot be empty"):
        retriever.search(None)  # type: ignore[arg-type]


def test_invalid_top_k_rejection(tmp_path: Path) -> None:
    """Verify 0, negative integers, booleans, and non-ints are rejected for top_k."""
    art_dir, corpus_file = _create_synthetic_artifact(tmp_path, num_docs=2)
    retriever = ContrieverRetriever.load(
        art_dir, corpus_file, encoder=FakeQueryEncoder(np.zeros(DEFAULT_EMBEDDING_DIM))
    )

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.search("query", top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.search("query", top_k=-3)

    # In Python, bool is a subclass of int; verify booleans are rejected
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.search("query", top_k=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.search("query", top_k=False)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.search("query", top_k="10")  # type: ignore[arg-type]


def test_duplicate_document_id_rejection(tmp_path: Path) -> None:
    """Verify duplicate document IDs raise ValueError."""
    art_dir, corpus_file = _create_synthetic_artifact(
        tmp_path, num_docs=3, duplicate_ids=True
    )
    with pytest.raises(ValueError, match="Duplicate document ID found"):
        ContrieverRetriever.load(art_dir, corpus_file)


def test_missing_document_id_rejection(tmp_path: Path) -> None:
    """Verify document IDs missing from the corpus raise ValueError."""
    art_dir, corpus_file = _create_synthetic_artifact(
        tmp_path, num_docs=3, missing_from_corpus=True
    )
    with pytest.raises(ValueError, match="not found in source corpus mapping"):
        ContrieverRetriever.load(art_dir, corpus_file)


def test_embedding_and_document_count_mismatch(tmp_path: Path) -> None:
    """Verify mismatch between embedding rows and document IDs raises ValueError."""
    art_dir, corpus_file = _create_synthetic_artifact(tmp_path, num_docs=3)

    # Alter document_ids.json to have only 2 items
    doc_ids_file = art_dir / "document_ids.json"
    doc_ids_file.write_text(json.dumps(["doc_0", "doc_1"]), encoding="utf-8")

    # Update checksum
    manifest_file = art_dir / "index_manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["document_ids_sha256"] = compute_sha256(doc_ids_file)
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Count mismatch"):
        ContrieverRetriever.load(art_dir, corpus_file)


def test_non_finite_embedding_rejection(tmp_path: Path) -> None:
    """Verify NaN or Inf values in embeddings matrix raise ValueError."""
    art_dir, corpus_file = _create_synthetic_artifact(
        tmp_path, num_docs=3, non_finite=True
    )
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        ContrieverRetriever.load(art_dir, corpus_file)


def test_incorrect_embedding_dimension_rejection(tmp_path: Path) -> None:
    """Verify non-768 embedding dimensions raise ValueError."""
    # 1. Manifest records non-768 dimension
    art_dir, corpus_file = _create_synthetic_artifact(
        tmp_path, num_docs=3, embedding_dim=512
    )
    with pytest.raises(ValueError, match="Manifest embedding dimension"):
        ContrieverRetriever.load(art_dir, corpus_file)

    # 2. Array shape mismatch against 768
    art_dir2, corpus_file2 = _create_synthetic_artifact(
        tmp_path / "case2", num_docs=3, embedding_dim=DEFAULT_EMBEDDING_DIM
    )
    bad_emb = np.zeros((3, 256), dtype=np.float32)
    np.save(str(art_dir2 / "passage_embeddings.npy"), bad_emb)
    # Update checksum so it passes checksum check and reaches dimension check
    manifest_file = art_dir2 / "index_manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["embedding_sha256"] = compute_sha256(art_dir2 / "passage_embeddings.npy")
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        ContrieverRetriever.load(art_dir2, corpus_file2)


def test_missing_artifact_files_raise_appropriate_errors(tmp_path: Path) -> None:
    """Verify FileNotFoundError is raised when directory or files are missing."""
    with pytest.raises(FileNotFoundError, match="Artifact directory not found"):
        ContrieverRetriever.load(
            tmp_path / "non_existent_dir", tmp_path / "corpus.jsonl"
        )

    art_dir, corpus_file = _create_synthetic_artifact(tmp_path, num_docs=2)
    # Remove one required file
    (art_dir / "passage_embeddings.npy").unlink()

    with pytest.raises(FileNotFoundError, match="Missing required artifact file"):
        ContrieverRetriever.load(art_dir, corpus_file)


def test_corrupt_manifest_raises_value_error(tmp_path: Path) -> None:
    """Verify malformed JSON manifest raises ValueError."""
    art_dir, corpus_file = _create_synthetic_artifact(tmp_path, num_docs=2)
    (art_dir / "index_manifest.json").write_text("{ corrupt json ...", encoding="utf-8")

    with pytest.raises(ValueError, match="Corrupt manifest"):
        ContrieverRetriever.load(art_dir, corpus_file)


def test_checksum_mismatches_raise_value_error(tmp_path: Path) -> None:
    """Verify corpus and embedding SHA-256 mismatches raise ValueError."""
    art_dir, corpus_file = _create_synthetic_artifact(tmp_path, num_docs=2)

    # 1. Tamper with corpus
    with open(corpus_file, "a", encoding="utf-8") as f:
        f.write('{"_id": "extra", "text": "extra"}\n')

    with pytest.raises(ValueError, match="Corpus checksum mismatch"):
        ContrieverRetriever.load(art_dir, corpus_file)

    # Restore corpus and tamper with embeddings
    art_dir2, corpus_file2 = _create_synthetic_artifact(tmp_path / "second", num_docs=2)
    emb_file = art_dir2 / "passage_embeddings.npy"
    with open(emb_file, "ab") as f:
        f.write(b"corruption")

    with pytest.raises(ValueError, match="Embeddings checksum mismatch"):
        ContrieverRetriever.load(art_dir2, corpus_file2)


def test_local_query_encoder_receives_expected_text(tmp_path: Path) -> None:
    """Verify search method passes stripped query string to encoder."""
    art_dir, corpus_file = _create_synthetic_artifact(tmp_path, num_docs=2)
    fake_encoder = FakeQueryEncoder(np.zeros(DEFAULT_EMBEDDING_DIM, dtype=np.float32))

    retriever = ContrieverRetriever.load(art_dir, corpus_file, encoder=fake_encoder)
    retriever.search("   Where was Marie Curie born?   ", top_k=2)

    assert fake_encoder.recorded_query == "Where was Marie Curie born?"


def test_missing_dense_dependencies_raises_informative_error() -> None:
    """Verify clear error instructions when torch / transformers are missing."""
    encoder = ContrieverEncoder()
    # In an environment without [dense] installed, raises ImportError
    with pytest.raises(ImportError, match=r'pip install -e "\.\[dense\]"'):
        encoder.encode_texts(["Hello world"])


def test_mean_pooling_excludes_padding_tokens() -> None:
    """Verify attention-mask-aware mean pooling excludes padding tokens."""
    # Batch size 1, 3 tokens, 2 hidden dims
    # Token 0: [2.0, 4.0], Token 1: [4.0, 6.0], Token 2 (padding): [100.0, 200.0]
    token_embeddings = np.array(
        [[[2.0, 4.0], [4.0, 6.0], [100.0, 200.0]]], dtype=np.float32
    )
    # Mask: tokens 0 and 1 valid (1), token 2 is padding (0)
    attention_mask = np.array([[1, 1, 0]], dtype=np.int64)

    pooled = mean_pooling(token_embeddings, attention_mask)
    # Expected mean of valid tokens only: [(2+4)/2, (4+6)/2] = [3.0, 5.0]
    assert pooled.shape == (1, 2)
    np.testing.assert_allclose(pooled[0], np.array([3.0, 5.0], dtype=np.float32))


def test_verify_artifact_script(tmp_path: Path) -> None:
    """Verify artifact script passes on valid artifact and fails on invalid."""
    art_dir, corpus_file = _create_synthetic_artifact(tmp_path, num_docs=4)

    metrics = verify_artifact(art_dir, corpus_file)
    assert metrics["passage_count"] == 4
    assert metrics["embedding_dim"] == DEFAULT_EMBEDDING_DIM
    assert metrics["model_id"] == DEFAULT_MODEL_ID
    assert metrics["model_revision"] == DEFAULT_MODEL_REVISION

    # Check failure on checksum mismatch
    with open(corpus_file, "a", encoding="utf-8") as f:
        f.write('{"_id": "modified", "text": "bad"}\n')

    with pytest.raises(ValueError, match="Source corpus SHA-256 mismatch"):
        verify_artifact(art_dir, corpus_file)


def test_existing_bm25_retrieval_unaffected(tmp_path: Path) -> None:
    """Verify previous BM25 index and retrieval functionality remains intact."""
    corpus_file = tmp_path / "bm25_sample_corpus.jsonl"
    doc1 = '{"_id": "doc1", "title": "Quantum", "text": "Quantum mechanics."}\n'
    doc2 = '{"_id": "doc2", "title": "Baking", "text": "Flour and sugar."}\n'
    corpus_file.write_text(doc1 + doc2, encoding="utf-8")
    bm25_dir = tmp_path / "bm25_test_index"
    retriever = BM25Retriever.build(corpus_file, bm25_dir)
    results = retriever.search("Quantum mechanics", top_k=2)

    assert len(results) >= 1
    assert results[0].document_id == "doc1"
    assert results[0].rank == 1
    assert results[0].score > 0.0
