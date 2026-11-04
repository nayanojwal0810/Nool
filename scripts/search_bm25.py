#!/usr/bin/env python3
"""Search a BM25 sparse index from the command line."""

import argparse
import sys

from nool.config import load_config, resolve_path
from nool.retrieval.bm25 import BM25Retriever


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Search passages using a saved BM25 index."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query string.",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Path to index directory (default: from configs/app.yaml).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of results to retrieve (default: from configs/app.yaml or 10).",
    )
    return parser.parse_args()


def truncate_text(text: str, max_chars: int = 150) -> str:
    """Shorten text for readable terminal preview."""
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + "..."


def main() -> None:
    """Execute search query."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    try:
        app_config = load_config()
    except Exception:
        app_config = {}

    bm25_cfg = app_config.get("retrieval", {}).get("bm25", {})

    index_str = args.index_dir or bm25_cfg.get(
        "index_dir", "indexes/hotpotqa_sample_bm25"
    )
    top_k = args.top_k or int(bm25_cfg.get("top_k", 10))

    index_path = resolve_path(index_str)

    if not index_path.exists():
        sys.stderr.write(
            f"Error: Index directory not found at: {index_path}\n"
            "Build the index first using: python scripts/build_bm25_index.py\n"
        )
        sys.exit(1)

    try:
        retriever = BM25Retriever.load(index_path)
    except FileNotFoundError as exc:
        sys.stderr.write(f"Error loading index: {exc}\n")
        sys.exit(1)
    except Exception as exc:
        sys.stderr.write(f"Failed to load index from {index_path}: {exc}\n")
        sys.exit(1)

    try:
        results = retriever.search(args.query, top_k=top_k)
    except ValueError as exc:
        sys.stderr.write(f"Search error: {exc}\n")
        sys.exit(1)

    print(f"\nQuery: {args.query}")
    print(f"Top {top_k} results from: {index_path}\n" + "-" * 70)

    if not results:
        print("No matching passages found for the query.")
        return

    for res in results:
        preview = truncate_text(res.text)
        title_str = f" [{res.title}]" if res.title else ""
        print(
            f"#{res.rank} | Score: {res.score:.4f} | ID: {res.document_id}{title_str}"
        )
        print(f"    Preview: {preview}\n")


if __name__ == "__main__":
    main()
