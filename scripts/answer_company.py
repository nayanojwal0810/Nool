#!/usr/bin/env python3
"""Answer a company question from current, authorized passages."""

import argparse
import os

from nool.config import resolve_path
from nool.data.documents import DocumentStore
from nool.generation.answers import CompanyAnswerService
from nool.generation.ollama import OllamaAnswerModel
from nool.retrieval.bm25 import BM25Retriever
from nool.retrieval.company import DEMO_USERS, CompanyHybridSearch
from nool.retrieval.contriever import ContrieverRetriever, compute_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--user", choices=sorted(DEMO_USERS), required=True)
    parser.add_argument("--database", default="data/processed/demo_documents.sqlite")
    parser.add_argument("--index", default="indexes/company")
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    root = resolve_path(args.index)
    store = DocumentStore(resolve_path(args.database))
    bm25 = BM25Retriever.load(root / "bm25")
    dense = ContrieverRetriever.load(root / "contriever", root / "corpus.jsonl")
    if bm25.manifest["source_corpus_sha256"] != compute_sha256(root / "corpus.jsonl"):
        raise RuntimeError("Company search indexes use different corpora")
    fingerprint = (root / "source_fingerprint.txt").read_text(encoding="utf-8")
    search = CompanyHybridSearch(store, bm25, dense, indexed_fingerprint=fingerprint)
    service = CompanyAnswerService(search, store, OllamaAnswerModel(model=args.model))
    result = service.answer(args.question, args.user)
    print(result.text)
    if result.citations:
        print("Sources:")
        for number, citation in enumerate(result.citations, start=1):
            print(f"{number}. {citation.label}")
            print(f'   "{citation.quote}"')


if __name__ == "__main__":
    main()
