#!/usr/bin/env python3
"""Execute BM25 vs Contriever retrieval benchmark evaluation on HotpotQA sample."""

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from typing import Any

from nool.config import load_config, resolve_path
from nool.evaluation.data import (
    BENCHMARK_CORPUS_SHA256,
    DEFAULT_SAMPLE_MANIFEST_PATH,
    compute_file_sha256,
    extract_artifact_configurations,
    load_and_validate_corpus,
    load_and_validate_qrels,
    load_and_validate_queries,
    load_sample_manifest,
    verify_retrievers_alignment,
)
from nool.evaluation.metrics import (
    compute_latency_statistics,
    compute_macro_averages,
    compute_query_metrics,
)
from nool.retrieval.bm25 import BM25Retriever
from nool.retrieval.contriever import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    ContrieverEncoder,
    ContrieverRetriever,
)


def get_git_commit() -> str | None:
    """Retrieve current Git commit SHA if in a repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for retrieval evaluation."""
    parser = argparse.ArgumentParser(
        description="Run retrieval benchmark evaluation: BM25 vs Contriever."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "dev"],
        help="Evaluation split to evaluate: 'test' (default) or 'dev'.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Retrieval rank cutoff (must be >= 10, default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write benchmark outputs (default: outputs/.../<split>).",
    )
    parser.add_argument(
        "--sample-manifest",
        type=str,
        default=DEFAULT_SAMPLE_MANIFEST_PATH,
        help="Path to sample manifest YAML (default: manifests/hotpotqa_sample.yaml).",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to source corpus JSONL (default: from configs/app.yaml).",
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        help="Path to queries JSONL (default: hotpotqa_sample/queries.jsonl).",
    )
    parser.add_argument(
        "--qrels",
        type=str,
        default=None,
        help="Path to qrels TSV (default: hotpotqa_sample/qrels/<split>.tsv).",
    )
    parser.add_argument(
        "--bm25-index",
        type=str,
        default=None,
        help="Path to BM25 index directory (default: from configs/app.yaml).",
    )
    parser.add_argument(
        "--contriever-artifact",
        type=str,
        default=None,
        help="Path to Contriever artifact directory (default: from configs/app.yaml).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device for query encoder: 'cpu', 'cuda', 'auto' (default: cpu).",
    )
    return parser.parse_args()


def main() -> None:
    """Run retrieval benchmark evaluation."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    # Cutoff validation
    if args.top_k < 10:
        sys.stderr.write(
            f"Error: --top-k must be at least 10 (got {args.top_k}) to compute "
            "Recall@10, MRR@10, nDCG@10, and MAP@10.\n"
        )
        sys.exit(1)

    # 1. Load application configuration
    try:
        app_config = load_config()
    except Exception as exc:
        sys.stderr.write(f"Configuration error: {exc}\n")
        sys.exit(1)

    bm25_cfg = app_config.get("retrieval", {}).get("bm25", {})
    contriever_cfg = app_config.get("retrieval", {}).get("contriever", {})

    split = args.split.lower().strip()
    corpus_str = args.corpus or bm25_cfg.get(
        "corpus_path", "data/processed/hotpotqa_sample/corpus.jsonl"
    )
    queries_str = args.queries or "data/processed/hotpotqa_sample/queries.jsonl"
    qrels_str = args.qrels or f"data/processed/hotpotqa_sample/qrels/{split}.tsv"
    bm25_index_str = args.bm25_index or bm25_cfg.get(
        "index_dir", "indexes/hotpotqa_sample_bm25"
    )
    contriever_art_str = args.contriever_artifact or contriever_cfg.get(
        "artifact_dir", "indexes/contriever_hotpotqa_sample"
    )
    device = args.device or contriever_cfg.get("device", "cpu")

    output_dir_str = args.output_dir or f"outputs/retrieval_benchmark/{split}"
    output_dir = resolve_path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = resolve_path(args.sample_manifest)
    corpus_path = resolve_path(corpus_str)
    queries_path = resolve_path(queries_str)
    qrels_path = resolve_path(qrels_str)
    bm25_index_dir = resolve_path(bm25_index_str)
    contriever_art_dir = resolve_path(contriever_art_str)

    # 2. Load and validate sample manifest
    try:
        sample_manifest = load_sample_manifest(manifest_path)
    except Exception as exc:
        sys.stderr.write(f"Sample manifest validation failed: {exc}\n")
        sys.exit(1)

    sample_info = sample_manifest["sample"]
    expected_count = sample_info["corpus_sample_size"]
    if split == "test":
        expected_q_count = sample_info["test_queries"]
        expected_qr_count = sample_info["qrels_counts"]["test"]
    elif split == "dev":
        expected_q_count = sample_info["development_queries"]
        expected_qr_count = sample_info["qrels_counts"]["dev"]
    else:
        sys.stderr.write(f"Unsupported split: {split}\n")
        sys.exit(1)

    # 3. Strict benchmark data validation
    print("=" * 80)
    print("Nool — BM25 vs Contriever Retrieval Benchmark Evaluation")
    print("=" * 80)
    print(f"Split:               {split}")
    print(f"Top K Cutoff:        {args.top_k}")
    print(f"Sample Manifest:     {manifest_path}")
    print(f"Corpus:              {corpus_path}")
    print(f"Queries:             {queries_path}")
    print(f"Qrels:               {qrels_path}")
    print(f"BM25 Index:          {bm25_index_dir}")
    print(f"Contriever Artifact: {contriever_art_dir}")
    print(f"Inference Device:    {device}")
    print(f"Output Directory:    {output_dir}")
    print("-" * 80)
    print("Validating benchmark data files against sample manifest...")

    is_standard_sample = args.corpus is None

    try:
        actual_corpus_sha = compute_file_sha256(corpus_path)
        if is_standard_sample and actual_corpus_sha != BENCHMARK_CORPUS_SHA256:
            raise ValueError(
                f"Standard benchmark corpus SHA-256 mismatch for {corpus_path}:\n"
                f"  Expected: {BENCHMARK_CORPUS_SHA256}\n"
                f"  Actual:   {actual_corpus_sha}"
            )

        expected_total_queries = (
            sample_info["development_queries"] + sample_info["test_queries"]
        )

        corpus_ids = load_and_validate_corpus(
            corpus_path,
            expected_sha256=None,
            expected_count=expected_count,
        )
        print(f"  ✓ Corpus verified: {len(corpus_ids):,} passages with unique IDs.")

        all_queries = load_and_validate_queries(
            queries_path,
            expected_count=expected_total_queries,
        )
        print(f"  ✓ Queries verified: {len(all_queries):,} unique non-empty queries.")

        corpus_ids_set = set(corpus_ids)
        queries_ids_set = set(all_queries.keys())

        qrels_by_query, query_order = load_and_validate_qrels(
            qrels_path,
            valid_query_ids=queries_ids_set,
            valid_corpus_ids=corpus_ids_set,
            expected_query_count=expected_q_count,
            expected_qrel_count=expected_qr_count,
        )
        total_qrels_count = sum(len(docs) for docs in qrels_by_query.values())
        print(
            f"  ✓ Qrels verified: {len(query_order)} {split} queries, "
            f"{total_qrels_count:,} positive judgments (matches manifest)."
        )
    except Exception as exc:
        sys.stderr.write(f"\nBenchmark Data Validation Failed: {exc}\n")
        sys.exit(1)

    # 4. Load retrieval systems and verify provenance
    print("-" * 80)
    print("Loading retrieval systems and verifying corpus provenance...")

    try:
        print("  - Loading BM25 retriever...")
        bm25_retriever = BM25Retriever.load(bm25_index_dir)
        print(f"    Loaded BM25 index ({len(bm25_retriever.documents):,} documents).")

        print("  - Loading Contriever retriever and CPU query encoder...")
        contriever_retriever = ContrieverRetriever.load(
            artifact_dir=contriever_art_dir,
            corpus_path=corpus_path,
            expected_passage_count=expected_count,
            expected_corpus_sha256=actual_corpus_sha,
        )
        model_id = contriever_retriever.manifest.get("model_id", DEFAULT_MODEL_ID)
        model_revision = contriever_retriever.manifest.get(
            "model_revision", DEFAULT_MODEL_REVISION
        )
        encoder = ContrieverEncoder(
            model_id=model_id,
            model_revision=model_revision,
            device=device,
        )
        contriever_retriever.encoder = encoder
        encoder._ensure_model_and_tokenizer()
        print(
            f"    Loaded Contriever retriever "
            f"({len(contriever_retriever.document_ids):,} passages)."
        )

        # Verify exact 1:1 ID alignment and BM25/Contriever corpus provenance
        verify_retrievers_alignment(
            bm25_retriever,
            contriever_retriever,
            expected_corpus_sha256=actual_corpus_sha,
            expected_count=expected_count,
        )
        print("  ✓ Verified BM25 and Contriever corpus provenance and 1:1 ID ordering.")

        # Extract and validate configuration metadata from manifests
        bm25_cfg_summary, contriever_cfg_summary = extract_artifact_configurations(
            bm25_retriever.manifest,
            contriever_retriever.manifest,
        )
    except Exception as exc:
        sys.stderr.write(f"\nRetriever Loading / Verification Failed: {exc}\n")
        sys.exit(1)

    # 5. Warm-up pass (excluded from latency and metrics)
    print("-" * 80)
    print("Executing warm-up pass (excluded from evaluation metrics and latency)...")
    warmup_query = "warmup query to initialize caches and tokenizers"
    bm25_retriever.search(warmup_query, top_k=args.top_k)
    contriever_retriever.search(warmup_query, top_k=args.top_k)
    print("  ✓ Warm-up complete.")

    # 6. Benchmark evaluation loop
    print("-" * 80)
    print(f"Evaluating {len(query_order)} {split} queries on BM25 and Contriever...")

    bm25_per_query: list[dict[str, Any]] = []
    contriever_per_query: list[dict[str, Any]] = []
    bm25_latencies: list[float] = []
    contriever_latencies: list[float] = []

    per_query_rows: list[dict[str, Any]] = []

    for idx, qid in enumerate(query_order, start=1):
        query_text = all_queries[qid]
        qrel_dict = qrels_by_query[qid]
        expected_rel_ids = sorted(
            [cid for cid, score in qrel_dict.items() if score > 0]
        )

        # Evaluate BM25
        t0_bm25 = time.perf_counter()
        bm25_results = bm25_retriever.search(query_text, top_k=args.top_k)
        bm25_latency_ms = (time.perf_counter() - t0_bm25) * 1000.0
        bm25_latencies.append(bm25_latency_ms)

        bm25_retrieved_ids = [r.document_id for r in bm25_results]
        bm25_ranks = [r.rank for r in bm25_results]
        bm25_metrics = compute_query_metrics(bm25_retrieved_ids, qrel_dict)
        bm25_per_query.append(bm25_metrics)

        # Evaluate Contriever
        t0_dense = time.perf_counter()
        dense_results = contriever_retriever.search(query_text, top_k=args.top_k)
        dense_latency_ms = (time.perf_counter() - t0_dense) * 1000.0
        contriever_latencies.append(dense_latency_ms)

        dense_retrieved_ids = [r.document_id for r in dense_results]
        dense_ranks = [r.rank for r in dense_results]
        dense_metrics = compute_query_metrics(dense_retrieved_ids, qrel_dict)
        contriever_per_query.append(dense_metrics)

        # Record per-query detail row
        per_query_rows.append(
            {
                "query_id": qid,
                "query_text": query_text,
                "relevant_count": len(expected_rel_ids),
                "expected_relevant_doc_ids": json.dumps(expected_rel_ids),
                "bm25_retrieved_ids": json.dumps(bm25_retrieved_ids),
                "bm25_retrieved_ranks": json.dumps(bm25_ranks),
                "contriever_retrieved_ids": json.dumps(dense_retrieved_ids),
                "contriever_retrieved_ranks": json.dumps(dense_ranks),
                "bm25_recall_1": round(bm25_metrics["recall@1"], 6),
                "bm25_recall_5": round(bm25_metrics["recall@5"], 6),
                "bm25_recall_10": round(bm25_metrics["recall@10"], 6),
                "bm25_mrr_10": round(bm25_metrics["mrr@10"], 6),
                "bm25_ndcg_10": round(bm25_metrics["ndcg@10"], 6),
                "bm25_ap_10": round(bm25_metrics["ap@10"], 6),
                "contriever_recall_1": round(dense_metrics["recall@1"], 6),
                "contriever_recall_5": round(dense_metrics["recall@5"], 6),
                "contriever_recall_10": round(dense_metrics["recall@10"], 6),
                "contriever_mrr_10": round(dense_metrics["mrr@10"], 6),
                "contriever_ndcg_10": round(dense_metrics["ndcg@10"], 6),
                "contriever_ap_10": round(dense_metrics["ap@10"], 6),
                "bm25_latency_ms": round(bm25_latency_ms, 3),
                "contriever_latency_ms": round(dense_latency_ms, 3),
            }
        )

        if idx % 100 == 0 or idx == len(query_order):
            print(f"  Processed {idx}/{len(query_order)} queries...")

    # 7. Aggregate metrics and latency statistics
    bm25_aggregates = compute_macro_averages(bm25_per_query)
    contriever_aggregates = compute_macro_averages(contriever_per_query)

    bm25_latency_stats = compute_latency_statistics(bm25_latencies)
    contriever_latency_stats = compute_latency_statistics(contriever_latencies)

    # 8. Write per_query_results.csv
    per_query_csv_path = output_dir / "per_query_results.csv"
    if per_query_rows:
        fieldnames = list(per_query_rows[0].keys())
        with open(per_query_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_query_rows)

    # 9. Write comparison.csv
    comparison_csv_path = output_dir / "comparison.csv"
    comparison_rows = [
        ("Recall@1", bm25_aggregates["recall@1"], contriever_aggregates["recall@1"]),
        ("Recall@5", bm25_aggregates["recall@5"], contriever_aggregates["recall@5"]),
        ("Recall@10", bm25_aggregates["recall@10"], contriever_aggregates["recall@10"]),
        ("MRR@10", bm25_aggregates["mrr@10"], contriever_aggregates["mrr@10"]),
        ("nDCG@10", bm25_aggregates["ndcg@10"], contriever_aggregates["ndcg@10"]),
        ("MAP@10", bm25_aggregates["map@10"], contriever_aggregates["map@10"]),
        (
            "Mean latency (ms)",
            bm25_latency_stats["mean_ms"],
            contriever_latency_stats["mean_ms"],
        ),
        (
            "Median latency (ms)",
            bm25_latency_stats["median_ms"],
            contriever_latency_stats["median_ms"],
        ),
        (
            "P95 latency (ms)",
            bm25_latency_stats["p95_ms"],
            contriever_latency_stats["p95_ms"],
        ),
    ]

    with open(comparison_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "BM25", "Contriever"])
        for metric_name, bm25_val, dense_val in comparison_rows:
            if "latency" in metric_name.lower():
                writer.writerow([metric_name, f"{bm25_val:.2f}", f"{dense_val:.2f}"])
            else:
                writer.writerow([metric_name, f"{bm25_val:.4f}", f"{dense_val:.4f}"])

    # 10. Write summary.json with actual manifest configurations
    summary_data = {
        "evaluation_split": split,
        "num_queries_evaluated": len(query_order),
        "num_qrels_rows": total_qrels_count,
        "corpus_size": len(corpus_ids),
        "corpus_sha256": actual_corpus_sha,
        "sample_manifest": {
            "source_path": str(manifest_path),
            "declared_corpus_sample_size": expected_count,
            "declared_queries": expected_q_count,
            "declared_qrels_count": expected_qr_count,
        },
        "bm25_configuration": bm25_cfg_summary,
        "contriever_configuration": contriever_cfg_summary,
        "device": device,
        "system_environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
        },
        "top_k": args.top_k,
        "latency_notes": {
            "bm25": (
                "Latency covers plaintext query tokenization and sparse lexical "
                "scoring."
            ),
            "contriever": (
                "Latency covers neural query tokenization, transformer forward pass on "
                "device, mean pooling, and exact matrix-vector dot product scoring."
            ),
        },
        "bm25_metrics": {
            "recall@1": round(bm25_aggregates["recall@1"], 6),
            "recall@5": round(bm25_aggregates["recall@5"], 6),
            "recall@10": round(bm25_aggregates["recall@10"], 6),
            "mrr@10": round(bm25_aggregates["mrr@10"], 6),
            "ndcg@10": round(bm25_aggregates["ndcg@10"], 6),
            "map@10": round(bm25_aggregates["map@10"], 6),
            "mean_latency_ms": round(bm25_latency_stats["mean_ms"], 3),
            "median_latency_ms": round(bm25_latency_stats["median_ms"], 3),
            "p95_latency_ms": round(bm25_latency_stats["p95_ms"], 3),
        },
        "contriever_metrics": {
            "recall@1": round(contriever_aggregates["recall@1"], 6),
            "recall@5": round(contriever_aggregates["recall@5"], 6),
            "recall@10": round(contriever_aggregates["recall@10"], 6),
            "mrr@10": round(contriever_aggregates["mrr@10"], 6),
            "ndcg@10": round(contriever_aggregates["ndcg@10"], 6),
            "map@10": round(contriever_aggregates["map@10"], 6),
            "mean_latency_ms": round(contriever_latency_stats["mean_ms"], 3),
            "median_latency_ms": round(contriever_latency_stats["median_ms"], 3),
            "p95_latency_ms": round(contriever_latency_stats["p95_ms"], 3),
        },
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
    }

    summary_json_path = output_dir / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 11. Console comparison output using dynamic split name
    split_title = split.capitalize()
    print("\n" + "=" * 80)
    print("Benchmark Evaluation Results")
    print("=" * 80)
    print(f"Evaluated {split_title} Queries:     {len(query_order)}")
    print(f"Total Relevance Judgments:  {total_qrels_count:,}")
    print(f"Corpus Passages:            {len(corpus_ids):,}")
    print("Validation Status:          PASSED")
    print(f"Evaluation Cutoff (Top K):  {args.top_k}")
    print("-" * 80)
    print(f"{'Metric':<24} {'BM25 (Sparse)':<24} {'Contriever (Dense)':<24}")
    print("-" * 80)
    for metric_name, bm25_val, dense_val in comparison_rows:
        if "latency" in metric_name.lower():
            bm25_str = f"{bm25_val:>10.2f} ms"
            dense_str = f"{dense_val:>10.2f} ms"
            print(f"{metric_name:<24} {bm25_str:<24} {dense_str:<24}")
        else:
            bm25_str = f"{bm25_val:>10.4f}"
            dense_str = f"{dense_val:>10.4f}"
            print(f"{metric_name:<24} {bm25_str:<24} {dense_str:<24}")
    print("-" * 80)
    print("Inspectable output files generated:")
    print(f"  - Summary:    {summary_json_path}")
    print(f"  - Comparison: {comparison_csv_path}")
    print(f"  - Per-Query:  {per_query_csv_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
