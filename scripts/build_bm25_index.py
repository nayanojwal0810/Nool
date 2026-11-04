#!/usr/bin/env python3
"""Build and persist a BM25 sparse index from a JSONL corpus."""

import argparse
import sys

from nool.config import load_config, resolve_path
from nool.retrieval.bm25 import BM25Retriever


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build BM25 index from a prepared corpus JSONL file."
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to source corpus JSONL (default: from configs/app.yaml).",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Path to destination index directory (default: from configs/app.yaml).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing index directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute index building."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    try:
        app_config = load_config()
    except Exception as exc:
        sys.stderr.write(f"Failed to load application configuration: {exc}\n")
        sys.exit(1)

    bm25_cfg = app_config.get("retrieval", {}).get("bm25", {})

    corpus_str = args.corpus or bm25_cfg.get(
        "corpus_path", "data/processed/hotpotqa_sample/corpus.jsonl"
    )
    index_str = args.index_dir or bm25_cfg.get(
        "index_dir", "indexes/hotpotqa_sample_bm25"
    )

    corpus_path = resolve_path(corpus_str)
    index_path = resolve_path(index_str)

    if not corpus_path.exists():
        sys.stderr.write(f"Error: Corpus file not found at: {corpus_path}\n")
        sys.exit(1)

    method = bm25_cfg.get("method", "lucene")
    k1 = float(bm25_cfg.get("k1", 1.5))
    b = float(bm25_cfg.get("b", 0.75))
    stopwords = bm25_cfg.get("stopwords", "english")
    stemming = bool(bm25_cfg.get("stemming", False))
    indexed_fields = tuple(bm25_cfg.get("indexed_fields", ["title", "text"]))

    print(f"Building BM25 index from: {corpus_path}")
    print(f"Destination directory: {index_path}")
    print(
        f"Settings: method={method}, k1={k1}, b={b}, "
        f"stopwords={stopwords}, stemming={stemming}, fields={indexed_fields}"
    )

    try:
        retriever = BM25Retriever.build(
            corpus_path=corpus_path,
            index_dir=index_path,
            method=method,
            k1=k1,
            b=b,
            stopwords=stopwords,
            stemmer=None if not stemming else "english",
            indexed_fields=indexed_fields,
            overwrite=args.overwrite,
        )
    except FileExistsError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)
    except Exception as exc:
        sys.stderr.write(f"Index build failed: {exc}\n")
        sys.exit(1)

    manifest = retriever.manifest
    print("BM25 index built successfully:")
    print(f" - Indexed documents: {manifest['document_count']}")
    print(f" - Source corpus filename: {manifest['source_corpus_filename']}")
    print(f" - Source corpus SHA-256: {manifest['source_corpus_sha256']}")
    print(
        f" - BM25 method: {manifest['method']} (k1={manifest['k1']}, b={manifest['b']})"
    )
    print(f" - Index directory: {index_path}")


if __name__ == "__main__":
    main()
