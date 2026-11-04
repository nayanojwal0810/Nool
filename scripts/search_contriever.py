#!/usr/bin/env python3
"""Execute dense semantic search using Contriever passage embeddings."""

import argparse
import sys

from nool.config import load_config, resolve_path
from nool.retrieval.contriever import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    ContrieverEncoder,
    ContrieverRetriever,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Search passages using Contriever dense retrieval."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query or question string.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of ranked passages to return (default: from configs/app.yaml).",
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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device: 'cpu', 'cuda', or 'auto' (default: cpu).",
    )
    return parser.parse_args()


def truncate_preview(text: str, max_chars: int = 160) -> str:
    """Format single-line text preview with ellipses."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


def main() -> None:
    """Execute Contriever search CLI."""
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

    if args.top_k is not None:
        top_k = args.top_k
    else:
        top_k = int(contriever_cfg.get("top_k", 10))

    device = args.device or contriever_cfg.get("device", "cpu")

    artifact_path = resolve_path(artifact_dir_str)
    corpus_path = resolve_path(corpus_str)

    if not artifact_path.exists():
        sys.stderr.write(
            f"Search error: Artifact directory not found at: {artifact_path}\n"
            "Please generate embeddings in Google Colab, download the ZIP artifact, "
            f"and extract it into: {artifact_path}\n"
        )
        sys.exit(1)

    if not corpus_path.exists():
        sys.stderr.write(f"Search error: Corpus file not found at: {corpus_path}\n")
        sys.exit(1)

    expected_count = 50000 if args.corpus is None else None
    expected_sha = (
        "c3970d91ebfd36d794066b42bd7acb85a744aabba2c97dd9a9bfde52d3dcbfbf"
        if args.corpus is None
        else None
    )

    # Load retriever and manifest with automatic validation
    try:
        retriever = ContrieverRetriever.load(
            artifact_dir=artifact_path,
            corpus_path=corpus_path,
            expected_passage_count=expected_count,
            expected_corpus_sha256=expected_sha,
        )
    except Exception as exc:
        sys.stderr.write(f"Failed to load Contriever artifact: {exc}\n")
        sys.exit(1)

    # Initialize query encoder using manifest's immutable model revision
    model_id = retriever.manifest.get("model_id", DEFAULT_MODEL_ID)
    model_revision = retriever.manifest.get("model_revision", DEFAULT_MODEL_REVISION)

    try:
        encoder = ContrieverEncoder(
            model_id=model_id,
            model_revision=model_revision,
            device=device,
        )
        retriever.encoder = encoder
    except Exception as exc:
        sys.stderr.write(f"Failed to initialize Contriever encoder: {exc}\n")
        sys.exit(1)

    # Execute search
    try:
        results = retriever.search(query=args.query, top_k=top_k)
    except Exception as exc:
        sys.stderr.write(f"Search error: {exc}\n")
        sys.exit(1)

    print(f"\nQuery: {args.query.strip()}")
    print(f"Top {len(results)} results from: {artifact_path}")
    print("-" * 70)

    if not results:
        print("No matching results found.")
        return

    for result in results:
        title_display = f" [{result.title}]" if result.title else ""
        preview = truncate_preview(result.text)
        doc_id_label = f"ID: {result.document_id}{title_display}"
        print(f"#{result.rank} | Score: {result.score:.4f} | {doc_id_label}")
        print(f"    Preview: {preview}\n")


if __name__ == "__main__":
    main()
