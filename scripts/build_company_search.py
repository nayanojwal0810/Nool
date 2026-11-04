#!/usr/bin/env python3
"""Build company BM25 and Contriever indexes from current documents."""

import argparse
import json

import numpy as np

from nool.config import resolve_path
from nool.data.documents import DocumentStore
from nool.retrieval.bm25 import BM25Retriever
from nool.retrieval.company import corpus_fingerprint, export_current_corpus
from nool.retrieval.contriever import (
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    ContrieverEncoder,
    compute_sha256,
    verify_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="data/processed/demo_documents.sqlite")
    parser.add_argument("--output", default="indexes/company")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    output = resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    store = DocumentStore(resolve_path(args.database))
    if not store.list_documents():
        parser.error("Document database is empty; ingest the collection first")
    fingerprint = corpus_fingerprint(store)
    corpus = output / "corpus.jsonl"
    count = export_current_corpus(store, corpus)
    if not count:
        parser.error("No active passages to index")

    BM25Retriever.build(corpus, output / "bm25")
    with corpus.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    encoder = ContrieverEncoder(
        model_id=DEFAULT_MODEL_ID,
        model_revision=DEFAULT_MODEL_REVISION,
        device=args.device,
    )
    texts = [f"{record['title']} {record['text']}" for record in records]
    embeddings = encoder.encode_texts(texts, batch_size=args.batch_size)
    dense = output / "contriever"
    dense.mkdir(exist_ok=True)
    np.save(dense / "passage_embeddings.npy", embeddings.astype(np.float32))
    (dense / "document_ids.json").write_text(
        json.dumps([record["_id"] for record in records]), encoding="utf-8"
    )
    manifest = {
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": DEFAULT_MODEL_REVISION,
        "embedding_dimension": 768,
        "pooling": "masked_mean",
        "max_seq_length": DEFAULT_MAX_SEQ_LENGTH,
        "normalize_embeddings": False,
        "similarity": "dot_product",
        "output_dtype": "float32",
        "source_corpus_sha256": compute_sha256(corpus),
        "embedding_sha256": compute_sha256(dense / "passage_embeddings.npy"),
        "document_ids_sha256": compute_sha256(dense / "document_ids.json"),
        "passage_count": count,
    }
    (dense / "index_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    verify_artifact(dense, corpus, expected_passage_count=count)
    if corpus_fingerprint(store) != fingerprint:
        raise RuntimeError("Documents changed during index build; rebuild the index")
    (output / "source_fingerprint.txt").write_text(fingerprint, encoding="utf-8")
    print(
        f"Indexed {count} current passages "
        f"from {len(store.list_documents())} documents."
    )


if __name__ == "__main__":
    main()
