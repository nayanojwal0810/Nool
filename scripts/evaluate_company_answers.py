#!/usr/bin/env python3
"""Save company answer predictions for review against prepared cases."""

import argparse
import json
import os

from nool.config import resolve_path
from nool.data.documents import DocumentStore
from nool.generation.answers import CompanyAnswerService
from nool.generation.ollama import OllamaAnswerModel
from nool.retrieval.bm25 import BM25Retriever
from nool.retrieval.company import CompanyHybridSearch
from nool.retrieval.contriever import (
    ContrieverEncoder,
    ContrieverRetriever,
    compute_sha256,
)

INITIAL_SCENARIOS = {
    "direct",
    "paraphrase",
    "multiple_sources",
    "near_miss",
    "access_denied",
    "unsupported",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="data/processed/demo_documents.sqlite")
    parser.add_argument("--index", default="indexes/company")
    parser.add_argument("--questions", default="data/demo/question_cases.jsonl")
    parser.add_argument("--output", default="outputs/company_answer_predictions.jsonl")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    index = resolve_path(args.index)
    store = DocumentStore(resolve_path(args.database))
    bm25 = BM25Retriever.load(index / "bm25")
    dense = ContrieverRetriever.load(
        index / "contriever",
        index / "corpus.jsonl",
        encoder=ContrieverEncoder(device=args.device),
    )
    if bm25.manifest["source_corpus_sha256"] != compute_sha256(index / "corpus.jsonl"):
        raise RuntimeError("Company search indexes use different corpora")
    fingerprint = (index / "source_fingerprint.txt").read_text(encoding="utf-8")
    search = CompanyHybridSearch(store, bm25, dense, indexed_fingerprint=fingerprint)
    service = CompanyAnswerService(search, store, OllamaAnswerModel(model=args.model))
    with resolve_path(args.questions).open(encoding="utf-8") as stream:
        cases = [json.loads(line) for line in stream]
    selected = [case for case in cases if case["scenario"] in INITIAL_SCENARIOS]
    if args.limit:
        selected = selected[: args.limit]

    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"answered": 0, "partial": 0, "unsupported": 0, "unavailable": 0}
    expected_citations = 0
    cited_all_expected = 0
    expected_refusals = 0
    refused = 0
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for case in selected:
            result = service.answer(case["question"], case["user_id"])
            counts[result.status] += 1
            source_coverage = None
            if case["scenario"] in {"access_denied", "unsupported"}:
                expected_refusals += 1
                refused += result.status == "unsupported"
            else:
                expected_citations += 1
                source_coverage = set(case["source_document_ids"]) <= {
                    citation.document_id for citation in result.citations
                }
                cited_all_expected += source_coverage
            record = {
                "case_id": case["case_id"],
                "scenario": case["scenario"],
                "question": case["question"],
                "expected_answer": case["expected_answer"],
                "expected_source_document_ids": case["source_document_ids"],
                "status": result.status,
                "answer": result.text,
                "expected_sources_cited": source_coverage,
                "citations": [
                    {
                        "document_id": citation.document_id,
                        "chunk_id": citation.chunk_id,
                        "label": citation.label,
                        "quote": citation.quote,
                    }
                    for citation in result.citations
                ],
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Reviewed {len(selected)} initial-state cases; saved {output}.")
    print(
        f"Answered: {counts['answered']}; partial: {counts['partial']}; "
        f"unsupported: {counts['unsupported']}; unavailable: {counts['unavailable']}."
    )
    print(
        f"All expected sources cited: {cited_all_expected}/{expected_citations}; "
        f"expected refusals: {refused}/{expected_refusals}."
    )


if __name__ == "__main__":
    main()
