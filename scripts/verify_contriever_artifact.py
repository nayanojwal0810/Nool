#!/usr/bin/env python3
"""Verify local Contriever passage embedding artifact and manifest alignment."""

import argparse
import sys

from nool.config import load_config, resolve_path
from nool.retrieval.contriever import verify_artifact


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify downloaded Contriever passage embedding artifact."
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Path to Contriever artifact directory (default: from configs/app.yaml).",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to source corpus JSONL (default: from configs/app.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    """Execute artifact verification CLI."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    try:
        app_config = load_config()
    except Exception as exc:
        sys.stderr.write(f"Configuration error: {exc}\n")
        sys.exit(1)

    contriever_cfg = app_config.get("retrieval", {}).get("contriever", {})

    artifact_dir_str = args.artifact_dir or contriever_cfg.get(
        "artifact_dir", "indexes/contriever_hotpotqa_sample"
    )
    corpus_str = args.corpus or contriever_cfg.get(
        "corpus_path", "data/processed/hotpotqa_sample/corpus.jsonl"
    )

    artifact_path = resolve_path(artifact_dir_str)
    corpus_path = resolve_path(corpus_str)

    print(f"Verifying Contriever artifact in: {artifact_path}")
    print(f"Against source corpus: {corpus_path}")

    try:
        metrics = verify_artifact(artifact_path, corpus_path)
    except Exception as exc:
        sys.stderr.write(f"\nVerification FAILED: {exc}\n")
        sys.exit(1)

    print("\nContriever Artifact Verification: PASSED")
    print("----------------------------------------------------------------------")
    print(f"Passage Count:       {metrics['passage_count']:,}")
    print(f"Embedding Dimension: {metrics['embedding_dim']}")
    print(f"Data Type:           {metrics['dtype']}")
    print(f"Model ID:            {metrics['model_id']}")
    print(f"Model Revision:      {metrics['model_revision']}")
    print(f"Pooling Method:      {metrics['pooling']}")
    print(f"Similarity Method:   {metrics['similarity']}")
    print(f"Corpus SHA-256:      {metrics['corpus_sha256']}")
    print(f"Embedding SHA-256:   {metrics['embedding_sha256']}")
    print(f"Doc IDs SHA-256:     {metrics['document_ids_sha256']}")
    print("----------------------------------------------------------------------")


if __name__ == "__main__":
    main()
