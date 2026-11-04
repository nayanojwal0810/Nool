"""Unit tests for retrieval evaluation metrics and benchmark data validation."""

import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nool.evaluation.data import (
    compute_file_sha256,
    load_and_validate_corpus,
    load_and_validate_qrels,
    load_and_validate_queries,
    verify_retrievers_alignment,
)
from nool.evaluation.metrics import (
    compute_ap_at_k,
    compute_latency_statistics,
    compute_macro_averages,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_query_metrics,
    compute_recall_at_k,
)

# ==============================================================================
# Metric Unit Tests
# ==============================================================================


def test_recall_at_k_single_and_multiple_relevant() -> None:
    """Test Recall@k with single and multiple ground truth documents."""
    retrieved = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
    relevant = {"doc_1", "doc_3"}

    # At k=1: only doc_1 is retrieved (1 / 2 = 0.5)
    assert compute_recall_at_k(retrieved, relevant, k=1) == pytest.approx(0.5)

    # At k=2: doc_1 is retrieved (1 / 2 = 0.5)
    assert compute_recall_at_k(retrieved, relevant, k=2) == pytest.approx(0.5)

    # At k=3: doc_1 and doc_3 are both retrieved (2 / 2 = 1.0)
    assert compute_recall_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)

    # At k=5: 2 / 2 = 1.0
    assert compute_recall_at_k(retrieved, relevant, k=5) == pytest.approx(1.0)


def test_recall_at_k_no_relevant_or_none_retrieved() -> None:
    """Test Recall@k when no relevant documents exist or none are retrieved."""
    retrieved = ["doc_a", "doc_b", "doc_c"]

    # When no relevant ground truth exists
    assert compute_recall_at_k(retrieved, set(), k=5) == 0.0

    # When relevant documents are not in retrieved list
    assert compute_recall_at_k(retrieved, {"doc_x", "doc_y"}, k=5) == 0.0


def test_recall_at_k_invalid_k() -> None:
    """Test that non-positive or boolean k raises ValueError."""
    with pytest.raises(ValueError, match="Cutoff k must be a positive integer"):
        compute_recall_at_k(["doc_1"], {"doc_1"}, k=0)

    with pytest.raises(ValueError, match="Cutoff k must be a positive integer"):
        compute_recall_at_k(["doc_1"], {"doc_1"}, k=-2)


def test_mrr_at_k_various_ranks() -> None:
    """Test MRR@k at various ranks and cutoff boundaries."""
    # Rank 1: 1 / 1 = 1.0
    assert compute_mrr_at_k(["d1", "d2", "d3"], {"d1"}, k=10) == pytest.approx(1.0)

    # Rank 2: 1 / 2 = 0.5
    assert compute_mrr_at_k(["d2", "d1", "d3"], {"d1"}, k=10) == pytest.approx(0.5)

    # Rank 10: 1 / 10 = 0.1
    top_10 = [f"neg_{i}" for i in range(9)] + ["d1"]
    assert compute_mrr_at_k(top_10, {"d1"}, k=10) == pytest.approx(0.1)

    # Rank 11: beyond cutoff 10 -> 0.0
    top_11 = [f"neg_{i}" for i in range(10)] + ["d1"]
    assert compute_mrr_at_k(top_11, {"d1"}, k=10) == 0.0

    # No relevant document retrieved
    assert compute_mrr_at_k(["neg_1", "neg_2"], {"d1"}, k=10) == 0.0


def test_mrr_at_k_multiple_relevant_takes_first() -> None:
    """Test MRR@k uses the earliest rank when multiple relevant docs exist."""
    retrieved = ["neg_0", "rel_b", "rel_a"]
    relevant = {"rel_a", "rel_b"}
    # rel_b is at rank 2, rel_a is at rank 3 -> MRR is 1 / 2 = 0.5
    assert compute_mrr_at_k(retrieved, relevant, k=10) == pytest.approx(0.5)


def test_ndcg_at_k_binary_relevance() -> None:
    """Test nDCG@k with standard binary relevance judgments."""
    qrels = {"doc_1": 1, "doc_2": 1}

    # Perfect ranking: both relevant documents at ranks 1 and 2
    perfect = ["doc_1", "doc_2", "doc_neg"]
    assert compute_ndcg_at_k(perfect, qrels, k=10) == pytest.approx(1.0)

    # Inverted ranking: doc_1 at rank 2, doc_2 at rank 3
    retrieved = ["doc_neg", "doc_1", "doc_2"]
    dcg = (1.0 / math.log2(3)) + (1.0 / math.log2(4))
    idcg = (1.0 / math.log2(2)) + (1.0 / math.log2(3))
    expected_ndcg = dcg / idcg
    assert compute_ndcg_at_k(retrieved, qrels, k=10) == pytest.approx(expected_ndcg)

    # No relevant retrieved
    assert compute_ndcg_at_k(["neg_1", "neg_2"], qrels, k=10) == 0.0


def test_ndcg_at_k_graded_relevance() -> None:
    """Test nDCG@k with graded relevance (scores > 1)."""
    # doc_high has score 3 (gain 2^3 - 1 = 7)
    # doc_low has score 1 (gain 2^1 - 1 = 1)
    qrels = {"doc_high": 3, "doc_low": 1}

    # Ideal order: doc_high at rank 1, doc_low at rank 2
    idcg = (7.0 / math.log2(2)) + (1.0 / math.log2(3))

    # Sub-optimal order: doc_low at rank 1, doc_high at rank 2
    retrieved = ["doc_low", "doc_high"]
    dcg = (1.0 / math.log2(2)) + (7.0 / math.log2(3))
    expected_ndcg = dcg / idcg

    assert compute_ndcg_at_k(retrieved, qrels, k=10) == pytest.approx(expected_ndcg)
    assert 0.0 < expected_ndcg < 1.0


def test_ndcg_at_k_cutoff_boundary() -> None:
    """Test that documents beyond cutoff k do not contribute to DCG."""
    qrels = {"doc_target": 1}
    # doc_target at rank 11
    retrieved = [f"neg_{i}" for i in range(10)] + ["doc_target"]
    assert compute_ndcg_at_k(retrieved, qrels, k=10) == 0.0


def test_ap_at_k_single_and_multiple_relevant() -> None:
    """Test AP@k computation against manually calculated precision steps."""
    # Case 1: 1 relevant document at rank 1
    assert compute_ap_at_k(["d1", "d2"], {"d1"}, k=10) == pytest.approx(1.0)

    # Case 2: 2 relevant documents: d1 at rank 1, d2 at rank 3
    # Rank 1: precision = 1 / 1 = 1.0
    # Rank 2: not relevant
    # Rank 3: precision = 2 / 3
    # Denominator: min(2, 10) = 2
    # AP = (1.0 + 2/3) / 2 = 5/6
    assert compute_ap_at_k(["d1", "d_neg", "d2"], {"d1", "d2"}, k=10) == pytest.approx(
        (1.0 + (2.0 / 3.0)) / 2.0
    )

    # Case 3: No relevant retrieved
    assert compute_ap_at_k(["d_neg1", "d_neg2"], {"d1"}, k=10) == 0.0

    # Case 4: Relevant at rank 11 (outside top 10)
    retrieved_11 = [f"neg_{i}" for i in range(10)] + ["d1"]
    assert compute_ap_at_k(retrieved_11, {"d1"}, k=10) == 0.0


def test_compute_query_metrics_bundle() -> None:
    """Test that compute_query_metrics returns all required metric keys."""
    qrels = {"d1": 1, "d2": 1}
    retrieved = ["d1", "d_neg", "d2"]
    metrics = compute_query_metrics(retrieved, qrels)

    assert "recall@1" in metrics
    assert "recall@5" in metrics
    assert "recall@10" in metrics
    assert "mrr@10" in metrics
    assert "ndcg@10" in metrics
    assert "ap@10" in metrics

    assert metrics["recall@1"] == pytest.approx(0.5)
    assert metrics["recall@5"] == pytest.approx(1.0)
    assert metrics["recall@10"] == pytest.approx(1.0)
    assert metrics["mrr@10"] == pytest.approx(1.0)


def test_macro_averages_includes_failed_queries() -> None:
    """Verify compute_macro_averages includes failed queries in the mean."""
    queries_metrics = [
        {"recall@10": 1.0, "ap@10": 1.0},
        {"recall@10": 0.5, "ap@10": 0.5},
        {"recall@10": 0.0, "ap@10": 0.0},  # failed query
    ]
    macro = compute_macro_averages(queries_metrics, metric_keys=("recall@10", "ap@10"))

    # Macro mean: (1.0 + 0.5 + 0.0) / 3 = 0.5
    assert macro["recall@10"] == pytest.approx(0.5)
    # ap@10 is reported as map@10
    assert macro["map@10"] == pytest.approx(0.5)


def test_latency_statistics() -> None:
    """Test mean, median, and p95 latency calculation."""
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = compute_latency_statistics(latencies)

    assert stats["mean_ms"] == pytest.approx(30.0)
    assert stats["median_ms"] == pytest.approx(30.0)
    assert stats["p95_ms"] == pytest.approx(48.0)

    empty_stats = compute_latency_statistics([])
    assert empty_stats["mean_ms"] == 0.0


# ==============================================================================
# Data Validation Unit Tests
# ==============================================================================


def test_load_and_validate_corpus(tmp_path: Path) -> None:
    """Test corpus file validation, checksums, and duplicate ID rejection."""
    corpus_file = tmp_path / "corpus.jsonl"
    corpus_file.write_text(
        '{"_id": "doc_1", "text": "First passage."}\n'
        '{"_id": "doc_2", "text": "Second passage."}\n',
        encoding="utf-8",
    )
    sha256 = compute_file_sha256(corpus_file)

    # Valid load
    ids = load_and_validate_corpus(
        corpus_file, expected_sha256=sha256, expected_count=2
    )
    assert ids == ["doc_1", "doc_2"]

    # Checksum mismatch
    with pytest.raises(ValueError, match="Corpus SHA-256 mismatch"):
        load_and_validate_corpus(
            corpus_file, expected_sha256="wrong_hash", expected_count=2
        )

    # Count mismatch
    with pytest.raises(ValueError, match="Corpus passage count mismatch"):
        load_and_validate_corpus(corpus_file, expected_sha256=sha256, expected_count=5)

    # Duplicate document ID
    dup_file = tmp_path / "dup_corpus.jsonl"
    dup_file.write_text(
        '{"_id": "doc_dup", "text": "Passage A."}\n'
        '{"_id": "doc_dup", "text": "Passage B."}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate document ID"):
        load_and_validate_corpus(dup_file, expected_sha256=None, expected_count=None)


def test_load_and_validate_queries(tmp_path: Path) -> None:
    """Test queries file validation, unique IDs, and empty text rejection."""
    queries_file = tmp_path / "queries.jsonl"
    queries_file.write_text(
        '{"_id": "q1", "text": "What is quantum computing?"}\n'
        '{"_id": "q2", "text": "How does photosynthesis work?"}\n',
        encoding="utf-8",
    )

    queries = load_and_validate_queries(queries_file)
    assert len(queries) == 2
    assert queries["q1"] == "What is quantum computing?"

    # Duplicate query ID
    dup_file = tmp_path / "dup_queries.jsonl"
    dup_file.write_text(
        '{"_id": "q1", "text": "Query text 1"}\n'
        '{"_id": "q1", "text": "Query text 2"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate query ID"):
        load_and_validate_queries(dup_file)

    # Malformed / empty query text
    blank_file = tmp_path / "blank_queries.jsonl"
    blank_file.write_text(
        '{"_id": "q_blank", "text": "   "}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Malformed or empty query text"):
        load_and_validate_queries(blank_file)


def test_load_and_validate_qrels(tmp_path: Path) -> None:
    """Test qrels validation, header checks, and duplicate rejection."""
    qrels_file = tmp_path / "test.tsv"
    qrels_file.write_text(
        "query-id\tcorpus-id\tscore\nq1\tdoc_1\t1\nq1\tdoc_2\t2\nq2\tdoc_2\t1\n",
        encoding="utf-8",
    )

    valid_queries = {"q1", "q2"}
    valid_corpus = {"doc_1", "doc_2"}

    qrels_map, q_order = load_and_validate_qrels(
        qrels_file,
        valid_query_ids=valid_queries,
        valid_corpus_ids=valid_corpus,
        expected_query_count=2,
        expected_qrel_count=3,
    )
    assert q_order == ["q1", "q2"]
    assert qrels_map["q1"]["doc_1"] == 1
    assert qrels_map["q1"]["doc_2"] == 2

    # Unknown query ID
    with pytest.raises(ValueError, match="does not exist in queries"):
        load_and_validate_qrels(
            qrels_file,
            valid_query_ids={"q1"},  # q2 missing
            valid_corpus_ids=valid_corpus,
            expected_query_count=None,
            expected_qrel_count=None,
        )

    # Unknown corpus ID
    with pytest.raises(ValueError, match="does not exist in corpus"):
        load_and_validate_qrels(
            qrels_file,
            valid_query_ids=valid_queries,
            valid_corpus_ids={"doc_1"},  # doc_2 missing
            expected_query_count=None,
            expected_qrel_count=None,
        )

    # Duplicate qrel pair
    dup_qrels = tmp_path / "dup_qrels.tsv"
    dup_qrels.write_text(
        "query-id\tcorpus-id\tscore\nq1\tdoc_1\t1\nq1\tdoc_1\t1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate qrel pair"):
        load_and_validate_qrels(
            dup_qrels,
            valid_query_ids=valid_queries,
            valid_corpus_ids=valid_corpus,
            expected_query_count=None,
            expected_qrel_count=None,
        )


def test_verify_retrievers_alignment() -> None:
    """Test verify_retrievers_alignment checks document count and ordering."""
    bm25_mock = MagicMock()
    contriever_mock = MagicMock()

    doc_mock_1 = MagicMock()
    doc_mock_1.id = "doc_a"
    doc_mock_2 = MagicMock()
    doc_mock_2.id = "doc_b"

    bm25_mock.documents = [doc_mock_1, doc_mock_2]
    contriever_mock.document_ids = ["doc_a", "doc_b"]

    # Aligned passes
    verify_retrievers_alignment(bm25_mock, contriever_mock, expected_count=2)

    # Count mismatch
    contriever_mock.document_ids = ["doc_a"]
    with pytest.raises(ValueError, match="Document count mismatch between retrievers"):
        verify_retrievers_alignment(bm25_mock, contriever_mock, expected_count=None)

    # Sequence mismatch
    contriever_mock.document_ids = ["doc_b", "doc_a"]
    with pytest.raises(ValueError, match="alignment mismatch at index 0"):
        verify_retrievers_alignment(bm25_mock, contriever_mock, expected_count=2)
