#!/usr/bin/env python3
"""Search current company documents for a demo user."""

import argparse

from nool.config import resolve_path
from nool.data.documents import DocumentStore
from nool.retrieval.bm25 import BM25Retriever
from nool.retrieval.company import DEMO_USERS, CompanyHybridSearch
from nool.retrieval.contriever import ContrieverRetriever, compute_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--user", choices=sorted(DEMO_USERS), required=True)
    parser.add_argument("--database", default="data/processed/demo_documents.sqlite")
    parser.add_argument("--index", default="indexes/company")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    root = resolve_path(args.index)
    store = DocumentStore(resolve_path(args.database))
    bm25 = BM25Retriever.load(root / "bm25")
    dense = ContrieverRetriever.load(root / "contriever", root / "corpus.jsonl")
    if bm25.manifest["source_corpus_sha256"] != compute_sha256(root / "corpus.jsonl"):
        raise RuntimeError("BM25 and Contriever indexes use different company corpora")
    fingerprint = (root / "source_fingerprint.txt").read_text(encoding="utf-8")
    search = CompanyHybridSearch(store, bm25, dense, indexed_fingerprint=fingerprint)
    for item in search.search(args.query, args.user, top_k=args.top_k):
        location = f"page {item.page_number}" if item.page_number else item.section
        print(
            f"{item.rank}. {item.title} "
            f"[{item.document_id}, {location}, v{item.version}]"
        )
        print(item.text)


if __name__ == "__main__":
    main()
