"""Retrieval and generation evaluation metrics."""

from nool.evaluation.data import (
    BENCHMARK_CORPUS_SHA256,
    BENCHMARK_CORPUS_SIZE,
    BENCHMARK_SPLIT_QRELS,
    BENCHMARK_SPLIT_QUERIES,
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

__all__ = [
    "BENCHMARK_CORPUS_SHA256",
    "BENCHMARK_CORPUS_SIZE",
    "BENCHMARK_SPLIT_QRELS",
    "BENCHMARK_SPLIT_QUERIES",
    "compute_ap_at_k",
    "compute_latency_statistics",
    "compute_macro_averages",
    "compute_mrr_at_k",
    "compute_ndcg_at_k",
    "compute_query_metrics",
    "compute_recall_at_k",
    "load_and_validate_corpus",
    "load_and_validate_qrels",
    "load_and_validate_queries",
    "verify_retrievers_alignment",
]
