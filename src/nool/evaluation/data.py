"""Benchmark data loading and validation for retrieval evaluation.

Validates:
- Benchmark sample manifest declaration and expected counts.
- Benchmark corpus checksum, document count, and string IDs.
- Queries file integrity, unique string IDs, and non-empty text.
- Qrels alignment with valid query IDs, corpus IDs, and scores.
- Qrels uniqueness (no duplicate (query_id, corpus_id) pairs).
- BM25 source corpus provenance checksum.
- Alignment of document IDs between BM25 and Contriever.
"""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from nool.retrieval.bm25 import BM25Retriever
from nool.retrieval.contriever import ContrieverRetriever

DEFAULT_SAMPLE_MANIFEST_PATH = "data/manifests/hotpotqa_sample.yaml"
BENCHMARK_CORPUS_SHA256 = (
    "c3970d91ebfd36d794066b42bd7acb85a744aabba2c97dd9a9bfde52d3dcbfbf"
)


def compute_file_sha256(file_path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 checksum of a file.

    Args:
        file_path: Path to target file.
        chunk_size: Read buffer size in bytes.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_sample_manifest(manifest_path: Path | str) -> dict[str, Any]:
    """Load and validate the benchmark sample manifest YAML file.

    Args:
        manifest_path: Path to hotpotqa_sample.yaml manifest.

    Returns:
        Parsed manifest dictionary.

    Raises:
        FileNotFoundError: If manifest_path does not exist.
        ValueError: If manifest is malformed, missing required fields, or has
            invalid types.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Sample manifest file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        raise ValueError(f"Malformed YAML in manifest {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Manifest root in {path} must be a dictionary.")

    sample = data.get("sample")
    if not isinstance(sample, dict):
        raise ValueError(f"Manifest {path} missing required 'sample' mapping.")

    required_int_fields = [
        "corpus_sample_size",
        "development_queries",
        "test_queries",
    ]
    for field_name in required_int_fields:
        val = sample.get(field_name)
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ValueError(
                f"Manifest field 'sample.{field_name}' in {path} must be a "
                f"positive integer, got {val!r}."
            )

    qrels_counts = sample.get("qrels_counts")
    if not isinstance(qrels_counts, dict):
        raise ValueError(
            f"Manifest field 'sample.qrels_counts' in {path} must be a dictionary."
        )

    for split_key in ["dev", "test"]:
        split_count = qrels_counts.get(split_key)
        if (
            not isinstance(split_count, int)
            or isinstance(split_count, bool)
            or split_count <= 0
        ):
            raise ValueError(
                f"Manifest 'sample.qrels_counts.{split_key}' in {path} must be a "
                f"positive integer, got {split_count!r}."
            )

    return data


def validate_bm25_provenance(
    bm25_retriever: BM25Retriever,
    expected_corpus_sha256: str,
) -> None:
    """Validate that BM25 index was built from the exact benchmark corpus.

    Args:
        bm25_retriever: Initialized BM25Retriever instance.
        expected_corpus_sha256: Expected SHA-256 checksum of source corpus.

    Raises:
        ValueError: If source_corpus_sha256 is missing, empty, or does not match
            expected_corpus_sha256.
    """
    manifest = getattr(bm25_retriever, "manifest", None)
    if not isinstance(manifest, dict):
        raise ValueError("BM25 retriever is missing manifest dictionary.")

    corpus_sha = manifest.get("source_corpus_sha256")
    if not isinstance(corpus_sha, str) or not corpus_sha.strip():
        raise ValueError(
            "BM25 index manifest is missing or has empty 'source_corpus_sha256' "
            "provenance checksum."
        )

    if corpus_sha != expected_corpus_sha256:
        raise ValueError(
            f"BM25 source corpus SHA-256 provenance mismatch:\n"
            f"  Expected: {expected_corpus_sha256}\n"
            f"  Manifest: {corpus_sha}"
        )


def extract_artifact_configurations(
    bm25_manifest: dict[str, Any],
    contriever_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract and strictly validate configuration metadata from index manifests.

    Args:
        bm25_manifest: Parsed BM25 index_manifest.json dictionary.
        contriever_manifest: Parsed Contriever index_manifest.json dictionary.

    Returns:
        Tuple of (bm25_config, contriever_config) dictionaries.

    Raises:
        ValueError: If manifests are not dictionaries or lack required schema fields.
    """
    if not isinstance(bm25_manifest, dict):
        raise ValueError("BM25 manifest must be a dictionary.")
    if not isinstance(contriever_manifest, dict):
        raise ValueError("Contriever manifest must be a dictionary.")

    required_bm25_fields = ["method", "k1", "b", "stopwords", "stemming"]
    for field_name in required_bm25_fields:
        if field_name not in bm25_manifest:
            raise ValueError(
                f"BM25 manifest missing required configuration field: '{field_name}'."
            )

    required_dense_fields = [
        "model_id",
        "model_revision",
        "pooling",
        "similarity_method",
    ]
    for field_name in required_dense_fields:
        if field_name not in contriever_manifest:
            raise ValueError(
                "Contriever manifest missing required configuration field: "
                f"'{field_name}'."
            )

    bm25_config = {
        "method": bm25_manifest["method"],
        "k1": bm25_manifest["k1"],
        "b": bm25_manifest["b"],
        "stopwords": bm25_manifest["stopwords"],
        "stemming": bm25_manifest["stemming"],
        "stemmer": bm25_manifest.get("stemmer"),
    }

    contriever_config = {
        "model_id": contriever_manifest["model_id"],
        "model_revision": contriever_manifest["model_revision"],
        "pooling": contriever_manifest["pooling"],
        "similarity_method": contriever_manifest["similarity_method"],
        "embedding_dimension": contriever_manifest.get("embedding_dimension", 768),
        "normalize_embeddings": contriever_manifest.get("normalize_embeddings", False),
    }

    return bm25_config, contriever_config


def load_and_validate_corpus(
    corpus_path: Path | str,
    expected_sha256: str | None = BENCHMARK_CORPUS_SHA256,
    expected_count: int | None = None,
) -> list[str]:
    """Validate and extract document IDs from the benchmark corpus.

    Strictly validates that document IDs are strings and preserves them exactly.

    Args:
        corpus_path: Path to corpus JSONL file.
        expected_sha256: Expected SHA-256 checksum (defaults to benchmark hash).
        expected_count: Expected document count (e.g. from manifest).

    Returns:
        List of document IDs in exact corpus order.

    Raises:
        FileNotFoundError: If corpus_path does not exist.
        ValueError: If SHA-256 mismatch, count mismatch, duplicate IDs, non-string
            IDs, or malformed JSON.
    """
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    if expected_sha256 is not None:
        actual_sha256 = compute_file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Corpus SHA-256 mismatch for {path}:\n"
                f"  Expected: {expected_sha256}\n"
                f"  Actual:   {actual_sha256}"
            )

    doc_ids: list[str] = []
    seen_ids: set[str] = set()

    with open(path, encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in corpus at {path}:{line_no}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(f"Corpus record at {path}:{line_no} must be a dict.")

            if "_id" not in record:
                raise ValueError(f"Missing '_id' field at {path}:{line_no}.")

            doc_id = record["_id"]
            if not isinstance(doc_id, str):
                raise ValueError(
                    f"Document '_id' at {path}:{line_no} must be a string, "
                    f"got {type(doc_id).__name__}."
                )

            if not doc_id.strip():
                raise ValueError(
                    f"Empty or blank '_id' field in corpus at {path}:{line_no}."
                )

            if doc_id in seen_ids:
                raise ValueError(
                    f"Duplicate document ID '{doc_id}' in corpus at {path}:{line_no}."
                )
            seen_ids.add(doc_id)

            text_val = record.get("text")
            if text_val is None or not isinstance(text_val, str):
                raise ValueError(f"Field 'text' at {path}:{line_no} must be a string.")

            doc_ids.append(doc_id)

    if expected_count is not None and len(doc_ids) != expected_count:
        raise ValueError(
            f"Corpus passage count mismatch for {path}: "
            f"got {len(doc_ids)}, expected exactly {expected_count}."
        )

    return doc_ids


def load_and_validate_queries(
    queries_path: Path | str,
) -> dict[str, str]:
    """Validate and load query records from queries.jsonl.

    Strictly validates that query IDs are strings and preserves them exactly.

    Args:
        queries_path: Path to queries JSONL file.

    Returns:
        Mapping of query ID to query text.

    Raises:
        FileNotFoundError: If queries_path does not exist.
        ValueError: If duplicate query IDs, non-string IDs, missing text, or blank
            queries are found.
    """
    path = Path(queries_path)
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path}")

    queries: dict[str, str] = {}

    with open(path, encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in queries at {path}:{line_no}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(f"Query record at {path}:{line_no} must be a dict.")

            if "_id" not in record:
                raise ValueError(f"Missing '_id' in queries at {path}:{line_no}.")

            qid = record["_id"]
            if not isinstance(qid, str):
                raise ValueError(
                    f"Query '_id' at {path}:{line_no} must be a string, "
                    f"got {type(qid).__name__}."
                )

            if not qid.strip():
                raise ValueError(
                    f"Empty or blank '_id' in queries at {path}:{line_no}."
                )

            if qid in queries:
                raise ValueError(
                    f"Duplicate query ID '{qid}' found in queries at {path}:{line_no}."
                )

            text = record.get("text")
            if text is None or not isinstance(text, str):
                raise ValueError(
                    f"Missing or non-string 'text' in query '{qid}' "
                    f"at {path}:{line_no}."
                )

            cleaned_text = text.strip()
            if not cleaned_text:
                raise ValueError(
                    f"Malformed or empty query text for query ID '{qid}' "
                    f"at {path}:{line_no}."
                )

            queries[qid] = cleaned_text

    return queries


def load_and_validate_qrels(
    qrels_path: Path | str,
    valid_query_ids: set[str],
    valid_corpus_ids: set[str],
    expected_query_count: int | None = None,
    expected_qrel_count: int | None = None,
) -> tuple[dict[str, dict[str, int]], list[str]]:
    """Validate and load relevance judgments from a TSV file.

    Args:
        qrels_path: Path to qrels TSV file (dev.tsv or test.tsv).
        valid_query_ids: Set of valid string query IDs loaded from queries.jsonl.
        valid_corpus_ids: Set of valid string document IDs loaded from corpus.jsonl.
        expected_query_count: Optional expected count of distinct queries in the split.
        expected_qrel_count: Optional expected total count of qrel rows in the split.

    Returns:
        Tuple of (qrels_by_query, query_order):
        - qrels_by_query: Mapping query_id -> {corpus_id: score}
        - query_order: List of unique query IDs in exact order of appearance in TSV.

    Raises:
        FileNotFoundError: If qrels_path does not exist.
        ValueError: If header is invalid, rows are malformed, query or corpus IDs
            are invalid/missing, duplicate pairs exist, or row/query counts mismatch.
    """
    path = Path(qrels_path)
    if not path.exists():
        raise FileNotFoundError(f"Qrels file not found: {path}")

    expected_header = ["query-id", "corpus-id", "score"]
    qrels_by_query: dict[str, dict[str, int]] = {}
    query_order: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()

    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty qrels file at {path}")

        normalized_header = [h.strip() for h in header]
        if normalized_header != expected_header:
            raise ValueError(
                f"Invalid qrels header in {path}: got {header}, "
                f"expected {expected_header}"
            )

        for line_no, row in enumerate(reader, 2):
            if not row or not any(row):
                continue

            if len(row) != 3:
                raise ValueError(
                    f"Malformed row at {path}:{line_no}: "
                    f"expected 3 columns, got {len(row)}"
                )

            raw_qid = row[0]
            raw_cid = row[1]
            raw_score = row[2]

            if not isinstance(raw_qid, str) or not raw_qid.strip():
                raise ValueError(f"Missing or blank query-id at {path}:{line_no}")
            if not isinstance(raw_cid, str) or not raw_cid.strip():
                raise ValueError(f"Missing or blank corpus-id at {path}:{line_no}")

            qid = raw_qid.strip()
            cid = raw_cid.strip()

            try:
                score = int(raw_score.strip())
            except ValueError as exc:
                raise ValueError(
                    f"Non-integer score '{raw_score}' at {path}:{line_no}"
                ) from exc

            # Validate ID existence against known string sets
            if qid not in valid_query_ids:
                raise ValueError(
                    f"Qrel query ID '{qid}' at {path}:{line_no} "
                    "does not exist in queries."
                )
            if cid not in valid_corpus_ids:
                raise ValueError(
                    f"Qrel corpus ID '{cid}' at {path}:{line_no} "
                    "does not exist in corpus."
                )

            # Reject duplicate query-document pairs
            pair = (qid, cid)
            if pair in seen_pairs:
                raise ValueError(
                    f"Duplicate qrel pair (query='{qid}', corpus='{cid}') "
                    f"at {path}:{line_no}."
                )
            seen_pairs.add(pair)

            if qid not in qrels_by_query:
                qrels_by_query[qid] = {}
                query_order.append(qid)

            qrels_by_query[qid][cid] = score

    total_rows = len(seen_pairs)
    total_queries = len(query_order)

    if expected_qrel_count is not None and total_rows != expected_qrel_count:
        raise ValueError(
            f"Qrels row count mismatch for {path}: got {total_rows}, "
            f"expected {expected_qrel_count}."
        )

    if expected_query_count is not None and total_queries != expected_query_count:
        raise ValueError(
            f"Unique query count mismatch in qrels for {path}: got {total_queries}, "
            f"expected {expected_query_count}."
        )

    # Ensure every evaluated query has at least one positive relevance judgment
    for qid in query_order:
        positive_rels = [cid for cid, s in qrels_by_query[qid].items() if s > 0]
        if not positive_rels:
            raise ValueError(
                f"Query ID '{qid}' in {path} has no relevant documents with score > 0."
            )

    return qrels_by_query, query_order


def verify_retrievers_alignment(
    bm25_retriever: BM25Retriever,
    contriever_retriever: ContrieverRetriever,
    expected_corpus_sha256: str | None = None,
    expected_count: int | None = None,
) -> None:
    """Verify that retrievers share the same corpus, provenance, and ID order.

    Args:
        bm25_retriever: Initialized BM25Retriever instance.
        contriever_retriever: Initialized ContrieverRetriever instance.
        expected_corpus_sha256: Optional expected corpus SHA-256 to verify provenance.
        expected_count: Expected document count (e.g. 50,000 from sample manifest).

    Raises:
        ValueError: If passage count mismatches, document ID sequences differ,
            or BM25/Contriever corpus provenance fails.
    """
    if expected_corpus_sha256 is not None:
        validate_bm25_provenance(bm25_retriever, expected_corpus_sha256)

        contriever_sha = contriever_retriever.manifest.get("source_corpus_sha256")
        if contriever_sha != expected_corpus_sha256:
            raise ValueError(
                f"Contriever source corpus SHA-256 provenance mismatch:\n"
                f"  Expected: {expected_corpus_sha256}\n"
                f"  Manifest: {contriever_sha}"
            )

    bm25_doc_ids = [doc.id for doc in bm25_retriever.documents]
    contriever_doc_ids = contriever_retriever.document_ids

    if expected_count is not None:
        if len(bm25_doc_ids) != expected_count:
            raise ValueError(
                f"BM25 document count is {len(bm25_doc_ids)}, "
                f"expected {expected_count}."
            )
        if len(contriever_doc_ids) != expected_count:
            raise ValueError(
                f"Contriever document count is {len(contriever_doc_ids)}, "
                f"expected {expected_count}."
            )

    if len(bm25_doc_ids) != len(contriever_doc_ids):
        raise ValueError(
            f"Document count mismatch between retrievers: BM25 has "
            f"{len(bm25_doc_ids)}, Contriever has {len(contriever_doc_ids)}."
        )

    if bm25_doc_ids != contriever_doc_ids:
        for idx, (bid, cid) in enumerate(
            zip(bm25_doc_ids, contriever_doc_ids, strict=False)
        ):
            if bid != cid:
                raise ValueError(
                    f"Retriever document ID alignment mismatch at index {idx}: "
                    f"BM25 has '{bid}', Contriever has '{cid}'."
                )
        raise ValueError("BM25 and Contriever document IDs do not match in order.")
