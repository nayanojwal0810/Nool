#!/usr/bin/env python3
"""Prepare a reproducible HotpotQA evaluation sample with preserved evidence."""

import argparse
import sys
from datetime import UTC, datetime

import yaml

from nool.config import resolve_path
from nool.data.hotpotqa import (
    HotpotQAError,
    compute_sha256,
)
from nool.data.sampling import (
    DEFAULT_CORPUS_SAMPLE_SIZE,
    DEFAULT_DEV_QUERY_COUNT,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TEST_QUERY_COUNT,
    prepare_hotpotqa_sample,
)

SAMPLING_PROCEDURE_DESC = (
    "Reproducible query selection from dev and test qrels with fixed random seed, "
    "guaranteed retention of all relevant positive passages, and streaming reservoir "
    "sampling of background/negative passages."
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build reproducible HotpotQA evaluation sample."
    )
    parser.add_argument(
        "--archive",
        type=str,
        default="data/raw/hotpotqa.zip",
        help="Path to raw hotpotqa.zip (default: data/raw/hotpotqa.zip).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/hotpotqa_sample",
        help="Output directory (default: data/processed/hotpotqa_sample).",
    )
    parser.add_argument(
        "--corpus-sample-size",
        type=int,
        default=DEFAULT_CORPUS_SAMPLE_SIZE,
        help=f"Corpus sample size (default: {DEFAULT_CORPUS_SAMPLE_SIZE}).",
    )
    parser.add_argument(
        "--dev-queries",
        type=int,
        default=DEFAULT_DEV_QUERY_COUNT,
        help=f"Number of development queries (default: {DEFAULT_DEV_QUERY_COUNT}).",
    )
    parser.add_argument(
        "--test-queries",
        type=int,
        default=DEFAULT_TEST_QUERY_COUNT,
        help=f"Number of test queries (default: {DEFAULT_TEST_QUERY_COUNT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for sampling (default: {DEFAULT_RANDOM_SEED}).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/hotpotqa_sample.yaml",
        help="Path for sample manifest (default: data/manifests/hotpotqa_sample.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    """Execute sample generation and manifest recording."""
    args = parse_args()
    archive_path = resolve_path(args.archive)
    output_dir = resolve_path(args.output_dir)
    manifest_path = resolve_path(args.manifest)

    if not archive_path.exists():
        sys.stderr.write(f"Error: Archive not found at {archive_path}\n")
        sys.exit(1)

    print(f"Reading archive: {archive_path}")
    print(
        f"Sampling parameters: corpus_size={args.corpus_sample_size}, "
        f"dev_queries={args.dev_queries}, test_queries={args.test_queries}, "
        f"seed={args.seed}"
    )

    try:
        result = prepare_hotpotqa_sample(
            archive_path=archive_path,
            output_dir=output_dir,
            corpus_sample_size=args.corpus_sample_size,
            dev_query_count=args.dev_queries,
            test_query_count=args.test_queries,
            seed=args.seed,
        )
    except (HotpotQAError, ValueError) as exc:
        sys.stderr.write(f"Sample preparation failed: {exc}\n")
        sys.exit(1)
    except Exception as exc:
        sys.stderr.write(f"Unexpected error: {exc}\n")
        sys.exit(1)

    archive_sha256 = compute_sha256(archive_path)
    generation_date = datetime.now(UTC).strftime("%Y-%m-%d")

    manifest_data = {
        "sample": {
            "source_archive_sha256": archive_sha256,
            "corpus_sample_size": result.corpus_sample_size,
            "development_queries": result.dev_query_count,
            "test_queries": result.test_query_count,
            "random_seed": result.random_seed,
            "sampling_procedure": SAMPLING_PROCEDURE_DESC,
            "positive_passages": result.positive_passage_count,
            "negative_passages": result.negative_passage_count,
            "qrels_counts": {
                "dev": result.dev_qrel_count,
                "test": result.test_qrel_count,
            },
            "generation_date": generation_date,
            "status": "verified",
        }
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(manifest_data, f, sort_keys=False)

    print("Sample generation complete:")
    print(f" - Output directory: {result.output_dir}")
    print(
        f" - Passages: {result.corpus_sample_size} "
        f"({result.positive_passage_count} positive, "
        f"{result.negative_passage_count} background/negative)"
    )
    print(f" - Dev queries: {result.dev_query_count} (qrels: {result.dev_qrel_count})")
    print(
        f" - Test queries: {result.test_query_count} (qrels: {result.test_qrel_count})"
    )
    print(f" - Manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()
