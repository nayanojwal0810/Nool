#!/usr/bin/env python3
"""Select RRF settings on dev, then evaluate the frozen setting on test."""

import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

from nool.evaluation.data import (
    BENCHMARK_CORPUS_SHA256,
    load_and_validate_corpus,
    load_and_validate_qrels,
    load_and_validate_queries,
    load_sample_manifest,
)
from nool.evaluation.metrics import compute_macro_averages, compute_query_metrics
from nool.retrieval.hybrid import reciprocal_rank_fusion


def load_baseline(split: str) -> tuple[list[dict], dict[str, dict[str, int]]]:
    """Check saved rankings against the fixed sample and the selected qrels split."""
    root = Path(__file__).resolve().parents[1]
    sample = load_sample_manifest(root / "data/manifests/hotpotqa_sample.yaml")[
        "sample"
    ]
    corpus_ids = load_and_validate_corpus(
        root / "data/processed/hotpotqa_sample/corpus.jsonl",
        expected_sha256=BENCHMARK_CORPUS_SHA256,
        expected_count=sample["corpus_sample_size"],
    )
    queries = load_and_validate_queries(
        root / "data/processed/hotpotqa_sample/queries.jsonl",
        expected_count=sample["development_queries"] + sample["test_queries"],
    )
    qrels, _ = load_and_validate_qrels(
        root / f"data/processed/hotpotqa_sample/qrels/{split}.tsv",
        valid_query_ids=set(queries),
        valid_corpus_ids=set(corpus_ids),
        expected_query_count=sample[
            "development_queries" if split == "dev" else "test_queries"
        ],
        expected_qrel_count=sample["qrels_counts"][split],
    )
    output = root / f"outputs/retrieval_benchmark/{split}"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("evaluation_split") != split
        or summary.get("corpus_sha256") != BENCHMARK_CORPUS_SHA256
        or summary.get("corpus_size") != len(corpus_ids)
        or summary.get("num_queries_evaluated") != len(qrels)
        or summary.get("top_k") != 10
    ):
        raise ValueError(
            f"Saved {split} benchmark summary does not match the fixed sample"
        )
    with (output / "per_query_results.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(qrels) or {r["query_id"] for r in rows} != set(qrels):
        raise ValueError(
            f"Saved {split} rankings do not cover exactly the qrels queries"
        )
    known_ids = set(corpus_ids)
    for row in rows:
        qid = row["query_id"]
        if row["query_text"] != queries[qid]:
            raise ValueError(f"Query text mismatch for {qid}")
        expected = {doc_id for doc_id, score in qrels[qid].items() if score > 0}
        if set(json.loads(row["expected_relevant_doc_ids"])) != expected:
            raise ValueError(f"Relevance IDs mismatch for {qid}")
        for source in ("bm25", "contriever"):
            ids = json.loads(row[f"{source}_retrieved_ids"])
            if len(ids) != len(set(ids)) or len(ids) > 10 or not set(ids) <= known_ids:
                raise ValueError(f"Invalid {source} ranking for {qid}")
    return rows, qrels


def evaluate(
    rows: list[dict], qrels: dict, rank_constant: int
) -> tuple[dict, list[dict]]:
    details = []
    for row in rows:
        bm25 = [
            SimpleNamespace(document_id=doc_id, title="", text="")
            for doc_id in json.loads(row["bm25_retrieved_ids"])
        ]
        dense = [
            SimpleNamespace(document_id=doc_id, title="", text="")
            for doc_id in json.loads(row["contriever_retrieved_ids"])
        ]
        start = time.perf_counter()
        fused = reciprocal_rank_fusion(
            bm25, dense, rank_constant=rank_constant, top_k=10
        )
        fusion_ms = (time.perf_counter() - start) * 1000
        ids = [result.document_id for result in fused]
        metrics = compute_query_metrics(ids, qrels[row["query_id"]])
        details.append(
            {
                "query_id": row["query_id"],
                "hybrid_ids": ids,
                "fusion_ms": fusion_ms,
                **metrics,
            }
        )
    return compute_macro_averages(details), details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs/hybrid_benchmark"
    out.mkdir(parents=True, exist_ok=True)
    rows, qrels = load_baseline(args.split)
    selection_path = out / "selection.json"
    if args.split == "dev":
        candidates = {}
        for rank_constant in (20, 60, 100):
            metrics, _ = evaluate(rows, qrels, rank_constant)
            candidates[rank_constant] = metrics
        # Recall first, then ranking quality; fixed tie-break avoids arbitrary choices.
        selected = max(
            candidates,
            key=lambda k: (candidates[k]["recall@10"], candidates[k]["ndcg@10"], -k),
        )
        selection_path.write_text(
            json.dumps(
                {
                    "selected_rank_constant": selected,
                    "candidate_depth": 10,
                    "selection_split": "dev",
                    "candidate_metrics": candidates,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if (
            selection.get("selection_split") != "dev"
            or selection.get("candidate_depth") != 10
        ):
            raise ValueError("Invalid development selection file")
        selected = selection["selected_rank_constant"]
    metrics, details = evaluate(rows, qrels, selected)
    baseline = {
        source: {
            key: sum(float(row[f"{source}_{column}"]) for row in rows) / len(rows)
            for key, column in (
                ("recall@10", "recall_10"),
                ("ndcg@10", "ndcg_10"),
                ("mrr@10", "mrr_10"),
                ("map@10", "ap_10"),
            )
        }
        for source in ("bm25", "contriever")
    }
    comparison = {
        "split": args.split,
        "queries": len(rows),
        "corpus_size": 50000,
        "candidate_depth": 10,
        "rank_constant": selected,
        "hybrid_metrics": metrics,
        "baseline_metrics": baseline,
        "latency_note": (
            "Hybrid retrieval runs both methods; saved baseline latencies plus "
            "fusion time are not an end-to-end timing measurement."
        ),
        "mean_fusion_ms": sum(d["fusion_ms"] for d in details) / len(details),
    }
    (out / f"{args.split}_summary.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    with (out / f"{args.split}_per_query.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "hybrid_ids",
                "recall@1",
                "recall@5",
                "recall@10",
                "mrr@10",
                "ndcg@10",
                "ap@10",
                "fusion_ms",
            ],
        )
        writer.writeheader()
        for detail in details:
            writer.writerow({**detail, "hybrid_ids": json.dumps(detail["hybrid_ids"])})
    print(f"Hybrid retrieval: {args.split} ({len(rows)} queries, 50,000 passages)")
    print(f"RRF rank constant: {selected}")
    for key in ("recall@10", "mrr@10", "ndcg@10", "map@10"):
        print(
            f"{key}: BM25 {baseline['bm25'][key]:.4f}, "
            f"Contriever {baseline['contriever'][key]:.4f}, "
            f"Hybrid {metrics[key]:.4f}"
        )
    print(f"Detailed results: {out / f'{args.split}_summary.json'}")


if __name__ == "__main__":
    main()
