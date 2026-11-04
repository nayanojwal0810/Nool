"""Information retrieval evaluation metrics.

This module provides standard IR ranking and accuracy metrics:
- Recall@k
- MRR@k (Mean Reciprocal Rank)
- nDCG@k (Normalized Discounted Cumulative Gain with graded relevance)
- AP@k / MAP@k (Average Precision / Mean Average Precision)
- Latency statistics (mean, median, p95)
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def compute_recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str] | dict[str, int] | Sequence[str],
    k: int,
) -> float:
    """Compute Recall@k for a single query.

    Recall@k is the fraction of relevant documents retrieved in the top k results:
        Recall@k = |retrieved[:k] ∩ relevant| / |relevant|

    Args:
        retrieved_ids: Ordered sequence of retrieved document IDs.
        relevant_ids: Set, dict, or sequence of ground-truth relevant document IDs.
        k: Evaluation rank cutoff (must be >= 1).

    Returns:
        Recall@k as a float between 0.0 and 1.0. If there are no relevant documents,
        returns 0.0.

    Raises:
        ValueError: If k is not a positive integer.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"Cutoff k must be a positive integer, got {k}")

    rel_set = set(relevant_ids) if not isinstance(relevant_ids, set) else relevant_ids
    if not rel_set:
        return 0.0

    top_k = retrieved_ids[:k]
    # Use distinct retrieved documents that are relevant
    distinct_retrieved_rel = {doc_id for doc_id in top_k if doc_id in rel_set}
    return float(len(distinct_retrieved_rel) / len(rel_set))


def compute_mrr_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str] | dict[str, int] | Sequence[str],
    k: int = 10,
) -> float:
    """Compute Reciprocal Rank at cutoff k for a single query.

    MRR@k is the reciprocal of the 1-based rank of the first relevant document
    retrieved within the first k positions:
        RR@k = 1 / rank_first_relevant (if rank <= k else 0.0)

    Args:
        retrieved_ids: Ordered sequence of retrieved document IDs.
        relevant_ids: Set, dict, or sequence of ground-truth relevant document IDs.
        k: Maximum rank cutoff (must be >= 1).

    Returns:
        Reciprocal rank as a float in [0.0, 1.0]. Returns 0.0 if no relevant
        document is found within top k or if relevant_ids is empty.

    Raises:
        ValueError: If k is not a positive integer.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"Cutoff k must be a positive integer, got {k}")

    rel_set = set(relevant_ids) if not isinstance(relevant_ids, set) else relevant_ids
    if not rel_set:
        return 0.0

    for rank_idx, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in rel_set:
            return float(1.0 / rank_idx)

    return 0.0


def compute_ndcg_at_k(
    retrieved_ids: Sequence[str],
    qrels: dict[str, int],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at cutoff k (nDCG@k).

    Uses graded relevance scores with standard formulation:
        Gain = 2^relevance - 1
        Discount = log2(rank + 1)
        DCG@k = sum_{i=1}^{min(k, |retrieved|)} Gain_i / Discount_i
        IDCG@k = DCG@k of ideally sorted ground-truth relevant documents
        nDCG@k = DCG@k / IDCG@k

    Args:
        retrieved_ids: Ordered sequence of retrieved document IDs.
        qrels: Mapping from document ID to integer relevance score.
        k: Maximum rank cutoff (must be >= 1).

    Returns:
        nDCG@k as a float in [0.0, 1.0]. Returns 0.0 if IDCG@k == 0.0.

    Raises:
        ValueError: If k is not a positive integer.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"Cutoff k must be a positive integer, got {k}")

    # Calculate DCG@k for retrieved ranking
    dcg = 0.0
    seen_docs: set[str] = set()
    for rank_idx, doc_id in enumerate(retrieved_ids[:k], start=1):
        # Ignore duplicate documents in retrieved list if any
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        score = qrels.get(doc_id, 0)
        if score > 0:
            gain = (2.0**score) - 1.0
            discount = math.log2(rank_idx + 1)
            dcg += gain / discount

    # Calculate Ideal DCG@k (IDCG@k) from ground-truth relevance scores
    positive_scores = sorted(
        [score for score in qrels.values() if score > 0],
        reverse=True,
    )
    if not positive_scores:
        return 0.0

    idcg = 0.0
    for rank_idx, score in enumerate(positive_scores[:k], start=1):
        gain = (2.0**score) - 1.0
        discount = math.log2(rank_idx + 1)
        idcg += gain / discount

    if idcg <= 0.0:
        return 0.0

    return float(dcg / idcg)


def compute_ap_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str] | dict[str, int] | Sequence[str],
    k: int = 10,
) -> float:
    """Compute Average Precision at cutoff k (AP@k) for a single query.

    Calculates precision at each rank containing a relevant document in the top k,
    divided by min(total_relevant_documents, k):
        AP@k = (sum_{i=1}^k Precision@i * is_relevant(doc_i)) / min(|relevant|, k)

    Args:
        retrieved_ids: Ordered sequence of retrieved document IDs.
        relevant_ids: Set, dict, or sequence of ground-truth relevant document IDs.
        k: Evaluation rank cutoff (must be >= 1).

    Returns:
        AP@k as a float in [0.0, 1.0]. Returns 0.0 if there are no relevant
        documents or none are retrieved in the top k.

    Raises:
        ValueError: If k is not a positive integer.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"Cutoff k must be a positive integer, got {k}")

    rel_set = set(relevant_ids) if not isinstance(relevant_ids, set) else relevant_ids
    total_rel = len(rel_set)
    if total_rel == 0:
        return 0.0

    denominator = min(total_rel, k)
    rel_count = 0
    precision_sum = 0.0
    seen_rel: set[str] = set()

    for rank_idx, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in rel_set and doc_id not in seen_rel:
            seen_rel.add(doc_id)
            rel_count += 1
            precision_at_i = rel_count / rank_idx
            precision_sum += precision_at_i

    return float(precision_sum / denominator)


def compute_query_metrics(
    retrieved_ids: Sequence[str],
    qrels: dict[str, int],
) -> dict[str, float]:
    """Compute all standard benchmark retrieval metrics for a single query.

    Args:
        retrieved_ids: Ranked list of retrieved document IDs.
        qrels: Mapping from document ID to integer relevance score.

    Returns:
        Dictionary mapping metric names to computed values:
        - recall@1
        - recall@5
        - recall@10
        - mrr@10
        - ndcg@10
        - ap@10
    """
    rel_ids = {doc_id for doc_id, score in qrels.items() if score > 0}
    return {
        "recall@1": compute_recall_at_k(retrieved_ids, rel_ids, k=1),
        "recall@5": compute_recall_at_k(retrieved_ids, rel_ids, k=5),
        "recall@10": compute_recall_at_k(retrieved_ids, rel_ids, k=10),
        "mrr@10": compute_mrr_at_k(retrieved_ids, rel_ids, k=10),
        "ndcg@10": compute_ndcg_at_k(retrieved_ids, qrels, k=10),
        "ap@10": compute_ap_at_k(retrieved_ids, rel_ids, k=10),
    }


def compute_macro_averages(
    per_query_metrics: Sequence[dict[str, Any]],
    metric_keys: Sequence[str] = (
        "recall@1",
        "recall@5",
        "recall@10",
        "mrr@10",
        "ndcg@10",
        "ap@10",
    ),
) -> dict[str, float]:
    """Compute macro-averages across all evaluated queries.

    The mean of per-query ap@10 is reported as map@10.

    Args:
        per_query_metrics: Sequence of per-query metric dictionaries.
        metric_keys: Keys to aggregate across queries.

    Returns:
        Dictionary mapping aggregate metric names (including map@10) to float means.
    """
    if not per_query_metrics:
        result = dict.fromkeys(metric_keys, 0.0)
        if "ap@10" in result:
            del result["ap@10"]
            result["map@10"] = 0.0
        return result

    total_queries = len(per_query_metrics)
    averages: dict[str, float] = {}

    for key in metric_keys:
        total_val = sum(float(item.get(key, 0.0)) for item in per_query_metrics)
        target_name = "map@10" if key == "ap@10" else key
        averages[target_name] = float(total_val / total_queries)

    return averages


def compute_latency_statistics(latencies_ms: Sequence[float]) -> dict[str, float]:
    """Compute summary latency statistics (mean, median, p95) in milliseconds.

    Args:
        latencies_ms: Sequence of latency measurements in milliseconds.

    Returns:
        Dictionary with keys 'mean_ms', 'median_ms', 'p95_ms'.
    """
    if not latencies_ms:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
        }

    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
    }
