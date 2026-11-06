#!/usr/bin/env python3
"""Save company answer predictions for review against prepared cases."""

import argparse
import json
import os
from collections import Counter

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
    parser.add_argument(
        "--output", default="outputs/company_answer_predictions_current.jsonl"
    )
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--case-ids", help="Comma-separated case IDs for a focused check"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.cpu_only and args.device != "cpu":
        parser.error("--cpu-only requires --device cpu")
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
    service = CompanyAnswerService(
        search,
        store,
        OllamaAnswerModel(
            model=args.model,
            timeout_seconds=args.timeout,
            cpu_only=args.cpu_only,
        ),
    )
    with resolve_path(args.questions).open(encoding="utf-8") as stream:
        cases = [json.loads(line) for line in stream]
    selected = [case for case in cases if case["scenario"] in INITIAL_SCENARIOS]
    if args.case_ids:
        requested = [case_id.strip() for case_id in args.case_ids.split(",")]
        if len(set(requested)) != len(requested) or not all(requested):
            parser.error("--case-ids must contain distinct, nonempty IDs")
        by_id = {case["case_id"]: case for case in selected}
        unknown = set(requested) - by_id.keys()
        if unknown:
            parser.error(f"Unknown case IDs: {', '.join(sorted(unknown))}")
        selected = [by_id[case_id] for case_id in requested]
    if args.limit:
        selected = selected[: args.limit]

    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if args.resume and output.is_file():
        with output.open(encoding="utf-8") as stream:
            existing = [json.loads(line) for line in stream]
        if len(existing) > len(selected) or any(
            record["case_id"] != case["case_id"]
            or record["question"] != case["question"]
            or record.get("model") != args.model
            or record.get("cpu_only", False) != args.cpu_only
            for record, case in zip(existing, selected, strict=False)
        ):
            raise ValueError("Saved predictions do not match the model or cases")
    mode = "a" if existing else "w"
    with output.open(mode, encoding="utf-8", newline="\n") as stream:
        for number, case in enumerate(
            selected[len(existing) :], start=len(existing) + 1
        ):
            result = service.answer(case["question"], case["user_id"])
            source_coverage = None
            if case["scenario"] not in {"access_denied", "unsupported"}:
                source_coverage = set(case["source_document_ids"]) <= {
                    citation.document_id for citation in result.citations
                }
            retrieved_coverage = None
            if case["scenario"] not in {"access_denied", "unsupported"}:
                retrieved_coverage = set(case["source_document_ids"]) <= set(
                    result.retrieved_document_ids
                )
            record = {
                "case_id": case["case_id"],
                "model": args.model,
                "cpu_only": args.cpu_only,
                "scenario": case["scenario"],
                "question": case["question"],
                "expected_answer": case["expected_answer"],
                "expected_source_document_ids": case["source_document_ids"],
                "status": result.status,
                "reason_code": result.reason_code,
                "answer": result.text,
                "retrieved_document_ids": result.retrieved_document_ids,
                "expected_sources_retrieved": retrieved_coverage,
                "evidence_chunk_ids": result.evidence_chunk_ids,
                "model_duration_ms": result.model_duration_ms,
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
            stream.flush()
            if number % 10 == 0 or number == len(selected):
                print(f"Completed {number}/{len(selected)} cases.", flush=True)
    with output.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    counts = Counter(record["status"] for record in records)
    supported = [
        record
        for record in records
        if record["scenario"] not in {"access_denied", "unsupported"}
    ]
    refusals = [
        record
        for record in records
        if record["scenario"] in {"access_denied", "unsupported"}
    ]
    cited_all_expected = sum(
        record["expected_sources_cited"] is True for record in supported
    )
    refused = sum(record["status"] == "unsupported" for record in refusals)
    print(f"Reviewed {len(selected)} initial-state cases; saved {output}.")
    print(
        f"Answered: {counts['answered']}; partial: {counts['partial']}; "
        f"unsupported: {counts['unsupported']}; unavailable: {counts['unavailable']}."
    )
    print(
        f"All expected sources cited: {cited_all_expected}/{len(supported)}; "
        f"expected refusals: {refused}/{len(refusals)}."
    )


if __name__ == "__main__":
    main()
