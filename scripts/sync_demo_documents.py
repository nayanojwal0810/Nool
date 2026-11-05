#!/usr/bin/env python3
"""Apply company document and access changes to the local document store."""

import argparse

from nool.config import resolve_path
from nool.data.collection import sync_collection
from nool.data.documents import DocumentStore
from nool.retrieval.company import index_matches_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/demo/manifest.yaml")
    parser.add_argument("--database", default="data/processed/demo_documents.sqlite")
    parser.add_argument("--index", default="indexes/company")
    args = parser.parse_args()
    database = resolve_path(args.database)
    result = sync_collection(resolve_path(args.manifest), database)
    print(
        f"Created: {len(result.created)}; content updated: "
        f"{len(result.content_updated)}; access changed: "
        f"{len(result.access_changed)}; archived: {len(result.archived)}; "
        f"reactivated: {len(result.reactivated)}; titles updated: "
        f"{len(result.title_updated)}."
    )
    if result.requires_rebuild or not index_matches_store(
        DocumentStore(database), resolve_path(args.index)
    ):
        print("Search index is outdated. Rebuild it on Colab GPU before searching.")
    else:
        print("Search index can remain in use.")


if __name__ == "__main__":
    main()
