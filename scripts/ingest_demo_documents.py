#!/usr/bin/env python3
"""Validate and ingest Lindenmere Services demo documents."""

import argparse

from nool.config import resolve_path
from nool.data.collection import ingest_collection, load_collection, validate_questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/demo/manifest.yaml")
    parser.add_argument("--database", default="data/processed/demo_documents.sqlite")
    args = parser.parse_args()
    manifest = resolve_path(args.manifest)
    _, documents = load_collection(manifest)
    questions = validate_questions(
        manifest.parent / "question_cases.jsonl",
        {entry["document_id"] for entry in documents},
    )
    count = ingest_collection(manifest, resolve_path(args.database))
    print(f"Ingested {count} documents; validated {len(questions)} question cases.")


if __name__ == "__main__":
    main()
