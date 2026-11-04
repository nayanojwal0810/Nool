"""Benchmark data loading and validation for retrieval evaluation.

Validates:
- Benchmark corpus checksum and unique document count.
- Queries file integrity, unique IDs, and non-empty text.
- Qrels alignment with valid query IDs and corpus IDs.
- Qrels uniqueness (no duplicate (query_id, corpus_id) pairs).
- Agreement with sample manifest counts.
- Alignment of document IDs between BM25 and Contriever.
"""

import csv
import hashlib
import json
from pathlib import Path

from nool.retrieval.bm25 import BM25Retriever
from nool.retrieval.contriever import ContrieverRetriever

BENCHMARK_CORPUS_SHA256 = (
    "c3970d91ebfd36d794066b42bd7acb85a744aabba2c97dd9a9bfde52d3dcbfbf"
)
BENCHMARK_CORPUS_SIZE = 50000
BENCHMARK_SPLIT_QUERIES = 500
BENCHMARK_SPLIT_QRELS = 1000


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


def load_and_validate_corpus(
    corpus_path: Path | str,
    expected_sha256: str | None = BENCHMARK_CORPUS_SHA256,
    expected_count: int | None = BENCHMARK_CORPUS_SIZE,
) -> list[str]:
    """Validate and extract document IDs from the benchmark corpus.

    Args:
        corpus_path: Path to corpus JSONL file.
        expected_sha256: Expected SHA-256 checksum (defaults to benchmark hash).
        expected_count: Expected document count (defaults to 50,000).

    Returns:
        List of document IDs in exact corpus order.

    Raises:
        FileNotFoundError: If corpus_path does not exist.
        ValueError: If SHA-256 mismatch, count mismatch, duplicate IDs, or malformed.
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

            doc_id = str(record.get("_id", "")).strip()
            if not doc_id:
                raise ValueError(f"Missing or empty '_id' field at {path}:{line_no}.")

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

    Args:
        queries_path: Path to queries JSONL file.

    Returns:
        Mapping of query ID to query text.

    Raises:
        FileNotFoundError: If queries_path does not exist.
        ValueError: If duplicate query IDs, missing text, or blank queries are found.
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

            qid = str(record.get("_id", "")).strip()
            if not qid:
                raise ValueError(
                    f"Missing or empty '_id' in queries at {path}:{line_no}."
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
    expected_query_count: int | None = BENCHMARK_SPLIT_QUERIES,
    expected_qrel_count: int | None = BENCHMARK_SPLIT_QRELS,
) -> tuple[dict[str, dict[str, int]], list[str]]:
    """Validate and load relevance judgments from a TSV file.

    Args:
        qrels_path: Path to qrels TSV file (dev.tsv or test.tsv).
        valid_query_ids: Set of valid query IDs loaded from queries.jsonl.
        valid_corpus_ids: Set of valid document IDs loaded from corpus.jsonl.
        expected_query_count: Expected count of distinct queries in the qrels split.
        expected_qrel_count: Expected total count of qrel rows in the qrels split.

    Returns:
        Tuple of (qrels_by_query, query_order):
        - qrels_by_query: Mapping query_id -> {corpus_id: score}
        - query_order: List of unique query IDs in exact order of appearance in TSV.

    Raises:
        FileNotFoundError: If qrels_path does not exist.
        ValueError: If header is invalid, rows are malformed, query or corpus IDs
            are missing, duplicate pairs exist, or row/query counts mismatch.
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

            qid = row[0].strip()
            cid = row[1].strip()
            score_str = row[2].strip()

            if not qid:
                raise ValueError(f"Missing query-id at {path}:{line_no}")
            if not cid:
                raise ValueError(f"Missing corpus-id at {path}:{line_no}")

            try:
                score = int(score_str)
            except ValueError as exc:
                raise ValueError(
                    f"Non-integer score '{score_str}' at {path}:{line_no}"
                ) from exc

            # Validate ID existence
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
    expected_count: int | None = BENCHMARK_CORPUS_SIZE,
) -> None:
    """Verify that BM25 and Contriever share the exact same corpus and ID order.

    Args:
        bm25_retriever: Initialized BM25Retriever instance.
        contriever_retriever: Initialized ContrieverRetriever instance.
        expected_count: Expected document count (50,000).

    Raises:
        ValueError: If passage count mismatches or document ID sequences differ.
    """
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
