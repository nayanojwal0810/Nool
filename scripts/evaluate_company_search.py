#!/usr/bin/env python3
"""Measure company source retrieval and restricted-source filtering."""

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from nool.config import resolve_path
from nool.data.documents import DocumentStore
from nool.retrieval.bm25 import BM25Retriever
from nool.retrieval.company import DEMO_USERS, CompanyHybridSearch, corpus_fingerprint
from nool.retrieval.contriever import ContrieverEncoder, ContrieverRetriever

ANSWERABLE = {"direct", "paraphrase", "multiple_sources", "near_miss"}


def distinct_documents(
    retriever, question: str, role: str, store: DocumentStore
) -> set[str]:
    found: list[str] = []
    count = (
        len(retriever.documents)
        if hasattr(retriever, "documents")
        else len(retriever.document_ids)
    )
    for hit in retriever.search(question, top_k=count):
        document_id = hit.metadata["document_id"]
        document = store.get_document(document_id)
        if (
            document.active
            and role in document.allowed_roles
            and document_id not in found
        ):
            found.append(document_id)
        if len(found) == 5:
            break
    return set(found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="data/processed/demo_documents.sqlite")
    parser.add_argument("--index", default="indexes/company")
    parser.add_argument("--questions", default="data/demo/question_cases.jsonl")
    parser.add_argument("--output", default="outputs/company_retrieval_evaluation.json")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    index = resolve_path(args.index)
    store = DocumentStore(resolve_path(args.database))
    bm25 = BM25Retriever.load(index / "bm25")
    dense = ContrieverRetriever.load(
        index / "contriever",
        index / "corpus.jsonl",
        encoder=ContrieverEncoder(device="cpu"),
    )
    fingerprint = (index / "source_fingerprint.txt").read_text(encoding="utf-8")
    if fingerprint != corpus_fingerprint(store):
        raise RuntimeError("Company search index is outdated; rebuild it")
    search = CompanyHybridSearch(store, bm25, dense, indexed_fingerprint=fingerprint)
    question_path = resolve_path(args.questions)
    with question_path.open(encoding="utf-8") as stream:
        cases = [json.loads(line) for line in stream]

    checked = Counter()
    passed = {name: Counter() for name in ("bm25", "contriever", "hybrid")}
    failures = []
    for case in cases:
        scenario = case["scenario"]
        if scenario not in ANSWERABLE:
            continue
        expected = set(case["source_document_ids"])
        role = DEMO_USERS[case["user_id"]]
        found = {
            "bm25": distinct_documents(bm25, case["question"], role, store),
            "contriever": distinct_documents(dense, case["question"], role, store),
            "hybrid": {
                hit.document_id
                for hit in search.search(case["question"], case["user_id"], top_k=5)
            },
        }
        checked[scenario] += 1
        for method, documents in found.items():
            if expected <= documents:
                passed[method][scenario] += 1
        if not expected <= found["hybrid"]:
            failures.append(case["case_id"])

    restricted_failures = []
    for case in cases:
        if case["scenario"] != "access_denied":
            continue
        forbidden = case["state_change"]["document_id"]
        found = {
            hit.document_id
            for hit in search.search(case["question"], case["user_id"], top_k=5)
        }
        if forbidden in found:
            restricted_failures.append(case["case_id"])

    permission_failures = []
    update_failures = []
    for case in cases:
        if case["scenario"] not in {"permission_change", "document_update"}:
            continue
        with tempfile.TemporaryDirectory() as directory:
            copy_path = Path(directory) / "documents.sqlite"
            shutil.copy2(store.database_path, copy_path)
            changed_store = DocumentStore(copy_path)
            change = case["state_change"]
            document_id = change["document_id"]
            if case["scenario"] == "permission_change":
                changed_store.set_allowed_roles(document_id, ["hr_admin"])
            else:
                document = changed_store.get_document(document_id)
                changed_store.ingest(
                    question_path.parent / change["replacement_source"],
                    document_id=document_id,
                    title=document.title,
                    owner=document.owner,
                    allowed_roles=document.allowed_roles,
                )
            changed_search = CompanyHybridSearch(
                changed_store,
                bm25,
                dense,
                indexed_fingerprint=fingerprint,
            )
            if case["scenario"] == "permission_change":
                found = {
                    hit.document_id
                    for hit in changed_search.search(
                        case["question"], case["user_id"], top_k=5
                    )
                }
                if document_id in found:
                    permission_failures.append(case["case_id"])
            else:
                try:
                    changed_search.search(case["question"], case["user_id"])
                except RuntimeError as exc:
                    if "outdated" not in str(exc):
                        update_failures.append(case["case_id"])
                else:
                    update_failures.append(case["case_id"])

    result = {
        "passages": len(bm25.documents),
        "metric": "all expected source documents in top five distinct documents",
        "checked": dict(checked),
        "passed": {name: dict(values) for name, values in passed.items()},
        "hybrid_failed_cases": failures,
        "restricted_source_leaks": restricted_failures,
        "permission_change_failures": permission_failures,
        "outdated_index_failures": update_failures,
    }
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    total = sum(checked.values())
    for method, values in passed.items():
        print(f"{method}: {sum(values.values())}/{total} source cases")
    print(f"Restricted-source leaks: {len(restricted_failures)}")
    print(f"Permission-change failures: {len(permission_failures)}")
    print(f"Outdated-index failures: {len(update_failures)}")
    print(f"Saved: {output}")
    if restricted_failures or permission_failures or update_failures:
        raise RuntimeError("Company search safety checks failed")


if __name__ == "__main__":
    main()
