"""Unit tests for BM25 sparse retrieval component."""

import json
from pathlib import Path
from unittest.mock import patch

import bm25s
import pytest

from nool.retrieval.bm25 import BM25Retriever, BM25SearchResult


def _write_corpus(path: Path, records: list[dict]) -> Path:
    """Helper writing a mock JSONL corpus file."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    """Create a small 5-passage synthetic corpus fixture."""
    docs = [
        {
            "_id": "doc1",
            "title": "Quantum Physics",
            "text": "Quantum mechanics explores phenomena at the atomic scale.",
            "metadata": {"category": "physics", "year": 1925},
        },
        {
            "_id": "doc2",
            "title": "Baking Bread",
            "text": "Sourdough bread requires flour, water, salt, and wild yeast.",
            "metadata": {"category": "cooking"},
        },
        {
            "_id": "doc3",
            "title": "Space Travel",
            "text": "Rockets use liquid oxygen and hydrogen for propulsion.",
            "metadata": {"category": "aerospace"},
        },
        {
            "_id": "doc4",
            "title": "Deep Sea Ecology",
            "text": "Hydrothermal vents support unique ecosystems in ocean trenches.",
            "metadata": {"category": "marine"},
        },
        {
            "_id": "doc5",
            "title": "Art History",
            "text": "The Renaissance movement flourished in 15th-century Florence.",
            "metadata": {"category": "arts"},
        },
    ]
    return _write_corpus(tmp_path / "test_corpus.jsonl", docs)


def test_build_and_load_index(sample_corpus: Path, tmp_path: Path) -> None:
    """Cover 1, 2, 4, 5: index creation, persistence, loading, and alignment."""
    index_dir = tmp_path / "test_index"

    retriever = BM25Retriever.build(
        corpus_path=sample_corpus,
        index_dir=index_dir,
        method="lucene",
        k1=1.5,
        b=0.75,
        stopwords="english",
    )

    assert index_dir.exists()
    assert (index_dir / "index_manifest.json").exists()
    assert (index_dir / "documents.jsonl").exists()
    assert (index_dir / "params.index.json").exists()
    assert retriever.manifest["document_count"] == 5
    assert retriever.manifest["method"] == "lucene"
    assert retriever.manifest["k1"] == 1.5
    assert retriever.manifest["b"] == 0.75

    # Test loading the saved index
    loaded = BM25Retriever.load(index_dir)
    assert len(loaded.documents) == 5
    assert loaded.documents[0].id == "doc1"
    assert loaded.documents[0].title == "Quantum Physics"
    assert loaded.documents[0].metadata == {"category": "physics", "year": 1925}
    assert loaded.documents[1].id == "doc2"
    assert loaded.documents[4].id == "doc5"


def test_search_relevance_and_fields(sample_corpus: Path, tmp_path: Path) -> None:
    """Cover 3, 4, 5, 6, 7: search returns passage with positive score and rank."""
    index_dir = tmp_path / "test_index"
    retriever = BM25Retriever.build(sample_corpus, index_dir)

    results = retriever.search("atomic mechanics", top_k=3)
    assert len(results) >= 1

    top = results[0]
    assert isinstance(top, BM25SearchResult)
    assert top.document_id == "doc1"
    assert top.title == "Quantum Physics"
    assert "atomic" in top.text
    assert top.rank == 1
    assert top.score > 0.0
    assert top.metadata["category"] == "physics"

    # Check scores in strictly descending order and 1-based sequential ranks
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score
        assert results[i].rank == i + 1


def test_top_k_limits_and_oversized_top_k(sample_corpus: Path, tmp_path: Path) -> None:
    """Cover 8, 9: top_k limits count, and top_k > corpus size handled safely."""
    index_dir = tmp_path / "test_index"
    retriever = BM25Retriever.build(sample_corpus, index_dir)

    # top_k = 1
    res1 = retriever.search("flour water bread", top_k=1)
    assert len(res1) == 1
    assert res1[0].document_id == "doc2"

    # top_k = 100 on 5-doc corpus
    res_large = retriever.search("flour water bread", top_k=100)
    assert 1 <= len(res_large) <= 5


def test_reject_blank_query_and_invalid_top_k(
    sample_corpus: Path, tmp_path: Path
) -> None:
    """Cover 10, 11: blank queries and non-positive/non-integer top_k rejected."""
    index_dir = tmp_path / "test_index"
    retriever = BM25Retriever.build(sample_corpus, index_dir)

    with pytest.raises(ValueError, match="empty or blank"):
        retriever.search("")

    with pytest.raises(ValueError, match="empty or blank"):
        retriever.search("   ")

    with pytest.raises(ValueError, match="positive integer"):
        retriever.search("bread", top_k=0)

    with pytest.raises(ValueError, match="positive integer"):
        retriever.search("bread", top_k=-5)

    with pytest.raises(ValueError, match="positive integer"):
        retriever.search("bread", top_k=True)  # bool is subclass of int in Python


def test_query_no_matching_terms_returns_empty(
    sample_corpus: Path, tmp_path: Path
) -> None:
    """Cover 12: queries with no matching terms return empty list."""
    index_dir = tmp_path / "test_index"
    retriever = BM25Retriever.build(sample_corpus, index_dir)

    # Word not in corpus
    results = retriever.search("xylophone zeppelin", top_k=5)
    assert results == []

    # Query with only stopwords
    results_stopwords = retriever.search("the and or is in", top_k=5)
    assert results_stopwords == []


def test_missing_corpus_file_raises(tmp_path: Path) -> None:
    """Cover 13: missing corpus file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Corpus file not found"):
        BM25Retriever.build(
            corpus_path=tmp_path / "nonexistent.jsonl",
            index_dir=tmp_path / "out",
        )


def test_malformed_json_raises_with_line_number(tmp_path: Path) -> None:
    """Cover 14: malformed JSON line reports line number."""
    bad_corpus = tmp_path / "bad.jsonl"
    with open(bad_corpus, "w", encoding="utf-8") as f:
        f.write('{"_id": "1", "text": "Valid line"}\n')
        f.write('{"_id": "2", text: broken json}\n')

    with pytest.raises(ValueError, match=r":2"):
        BM25Retriever.build(bad_corpus, tmp_path / "out")


def test_missing_fields_and_duplicates_rejected(tmp_path: Path) -> None:
    """Cover 15: missing _id, text, non-dict record, and duplicate IDs rejected."""
    # Non-dict record
    p1 = _write_corpus(tmp_path / "c1.jsonl", ["string instead of dict"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="JSON object"):
        BM25Retriever.build(p1, tmp_path / "out1")

    # Missing _id
    p2 = _write_corpus(tmp_path / "c2.jsonl", [{"text": "Missing id"}])
    with pytest.raises(ValueError, match="'_id'"):
        BM25Retriever.build(p2, tmp_path / "out2")

    # Missing text
    p3 = _write_corpus(tmp_path / "c3.jsonl", [{"_id": "doc1"}])
    with pytest.raises(ValueError, match="'text'"):
        BM25Retriever.build(p3, tmp_path / "out3")

    # Duplicate IDs
    p4 = _write_corpus(
        tmp_path / "c4.jsonl",
        [{"_id": "dup", "text": "First"}, {"_id": "dup", "text": "Second"}],
    )
    with pytest.raises(ValueError, match="Duplicate document ID 'dup'"):
        BM25Retriever.build(p4, tmp_path / "out4")


def test_empty_corpus_rejected(tmp_path: Path) -> None:
    """Cover 16: empty corpus raises ValueError."""
    empty_corpus = tmp_path / "empty.jsonl"
    empty_corpus.write_text("\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contains no valid records"):
        BM25Retriever.build(empty_corpus, tmp_path / "out")


def test_existing_index_not_overwritten_without_permission(
    sample_corpus: Path, tmp_path: Path
) -> None:
    """Cover 17: existing non-empty index dir raises error unless overwrite=True."""
    index_dir = tmp_path / "existing_index"
    BM25Retriever.build(sample_corpus, index_dir)

    with pytest.raises(FileExistsError, match="already exists"):
        BM25Retriever.build(sample_corpus, index_dir, overwrite=False)

    # Succeeded with overwrite=True
    rebuilt = BM25Retriever.build(sample_corpus, index_dir, overwrite=True)
    assert len(rebuilt.documents) == 5


def test_missing_or_incomplete_index_files_raise(tmp_path: Path) -> None:
    """Cover 18: missing index dir or missing index files produce clear error."""
    with pytest.raises(FileNotFoundError, match="Index directory not found"):
        BM25Retriever.load(tmp_path / "missing_dir")

    # Incomplete index dir (missing manifest)
    incomplete_dir = tmp_path / "incomplete"
    incomplete_dir.mkdir()
    (incomplete_dir / "documents.jsonl").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Incomplete BM25 index"):
        BM25Retriever.load(incomplete_dir)


def test_save_load_search_lifecycle(sample_corpus: Path, tmp_path: Path) -> None:
    """End-to-end lifecycle: build, save, load in a new instance, and search."""
    index_dir = tmp_path / "lifecycle_index"

    # 1. Build and save
    BM25Retriever.build(sample_corpus, index_dir)

    # 2. Load fresh instance
    loaded_retriever = BM25Retriever.load(index_dir)

    # 3. Search using newly loaded retriever
    results = loaded_retriever.search("Quantum mechanics atomic scale", top_k=3)

    assert len(results) >= 1
    top = results[0]
    assert top.document_id == "doc1"
    assert top.title == "Quantum Physics"
    assert top.score > 0.0
    assert top.rank == 1
    assert "atomic" in top.text


def test_stemmer_setting_consistency_between_indexing_and_search(
    tmp_path: Path,
) -> None:
    """Verify stemmer setting is stored in manifest, restored, and used in search."""
    docs = [
        {
            "_id": "doc1",
            "title": "Network Cables",
            "text": "The technician connects the optical cables.",
        },
        {
            "_id": "doc2",
            "title": "Baking",
            "text": "Flour and water make dough.",
        },
    ]
    corpus_file = _write_corpus(tmp_path / "stem_corpus.jsonl", docs)
    stem_index_dir = tmp_path / "stem_index"
    no_stem_index_dir = tmp_path / "no_stem_index"

    # 1. Build with stemming enabled ("english")
    BM25Retriever.build(corpus_file, stem_index_dir, stemmer="english")
    manifest = json.loads(
        (stem_index_dir / "index_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["stemmer"] == "english"
    assert manifest["stemming"] is True

    # Load in a fresh retriever
    stemmed_retriever = BM25Retriever.load(stem_index_dir)
    assert stemmed_retriever.stemmer is not None

    # Search with inflected word 'connecting':
    # With stemming, 'connecting' stems to 'connect', matching 'connects' in doc1
    with patch("bm25s.tokenize", wraps=bm25s.tokenize) as mock_tokenize:
        stemmed_results = stemmed_retriever.search("connecting", top_k=2)
        assert len(stemmed_results) == 1
        assert stemmed_results[0].document_id == "doc1"
        assert mock_tokenize.call_count == 1
        _, kwargs = mock_tokenize.call_args
        assert kwargs["stemmer"] is stemmed_retriever.stemmer
        assert kwargs["stemmer"] is not None

    # 2. Build with stemming disabled (default: None)
    BM25Retriever.build(corpus_file, no_stem_index_dir, stemmer=None)
    manifest_no_stem = json.loads(
        (no_stem_index_dir / "index_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_no_stem["stemmer"] is None
    assert manifest_no_stem["stemming"] is False

    no_stem_retriever = BM25Retriever.load(no_stem_index_dir)
    assert no_stem_retriever.stemmer is None

    # Search with 'connecting' when stemming is disabled:
    # 'connecting' does not match 'connects', yielding 0 results
    with patch("bm25s.tokenize", wraps=bm25s.tokenize) as mock_tokenize_none:
        no_stem_results = no_stem_retriever.search("connecting", top_k=2)
        assert no_stem_results == []
        assert mock_tokenize_none.call_count == 1
        _, kwargs_none = mock_tokenize_none.call_args
        assert kwargs_none["stemmer"] is None
