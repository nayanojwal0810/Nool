"""HotpotQA dataset reader, validator, and auditor."""

import csv
import hashlib
import io
import json
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class HotpotQAError(Exception):
    """Base exception for HotpotQA dataset operations."""


class HotpotQAMissingMemberError(HotpotQAError):
    """Raised when an expected archive member is missing."""


class HotpotQADataError(HotpotQAError):
    """Raised when a record in the dataset is malformed or invalid."""

    def __init__(self, message: str, file_name: str, line_number: int) -> None:
        super().__init__(f"{message} (file: {file_name}, line: {line_number})")
        self.file_name = file_name
        self.line_number = line_number


@dataclass(frozen=True)
class CorpusRecord:
    """A single document passage from the HotpotQA corpus."""

    id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to BEIR JSONL format dictionary."""
        result: dict[str, Any] = {
            "_id": self.id,
            "title": self.title,
            "text": self.text,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass(frozen=True)
class QueryRecord:
    """A single query record from HotpotQA."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to BEIR JSONL format dictionary."""
        result: dict[str, Any] = {
            "_id": self.id,
            "text": self.text,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass(frozen=True)
class QrelRecord:
    """A relevance judgment linking a query to a relevant passage."""

    query_id: str
    corpus_id: str
    score: int


@contextmanager
def _open_archive(source: Path | str | zipfile.ZipFile) -> Iterator[zipfile.ZipFile]:
    """Safely provide an open ZipFile instance without leaking resources."""
    if isinstance(source, zipfile.ZipFile):
        yield source
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Archive file not found: {path}")
        if not zipfile.is_zipfile(path):
            raise HotpotQAError(f"Not a valid ZIP archive: {path}")
        with zipfile.ZipFile(path, "r") as zf:
            yield zf


def compute_sha256(file_path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Calculate the SHA-256 checksum of a file.

    Args:
        file_path: Path to the target file.
        chunk_size: Read buffer size in bytes.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def iter_corpus(
    source: Path | str | zipfile.ZipFile,
    member_name: str = "hotpotqa/corpus.jsonl",
) -> Iterator[CorpusRecord]:
    """Stream corpus records incrementally from an archive.

    Args:
        source: Archive path or open ZipFile object.
        member_name: Archive path for corpus JSONL.

    Yields:
        Validated CorpusRecord instances.

    Raises:
        HotpotQAMissingMemberError: If member_name does not exist in the archive.
        HotpotQADataError: If any line contains invalid JSON or lacks required fields.
    """
    with _open_archive(source) as zf:
        if member_name not in zf.namelist():
            raise HotpotQAMissingMemberError(f"Missing archive member: {member_name}")
        with zf.open(member_name, "r") as raw_file:
            for line_no, raw_line in enumerate(raw_file, 1):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise HotpotQADataError(
                        f"Malformed JSON: {exc}",
                        file_name=member_name,
                        line_number=line_no,
                    ) from exc

                if not isinstance(item, dict):
                    raise HotpotQADataError(
                        "Record must be a JSON object",
                        file_name=member_name,
                        line_number=line_no,
                    )

                doc_id = str(item.get("_id", "")).strip()
                text = item.get("text")
                if not doc_id:
                    raise HotpotQADataError(
                        "Missing or empty '_id' field",
                        file_name=member_name,
                        line_number=line_no,
                    )
                if text is None or not isinstance(text, str):
                    raise HotpotQADataError(
                        "Missing or non-string 'text' field",
                        file_name=member_name,
                        line_number=line_no,
                    )

                title = item.get("title")
                title_str = str(title) if title is not None else ""
                metadata = item.get("metadata")
                meta_dict = metadata if isinstance(metadata, dict) else {}

                yield CorpusRecord(
                    id=doc_id,
                    text=text,
                    title=title_str,
                    metadata=meta_dict,
                )


def iter_queries(
    source: Path | str | zipfile.ZipFile,
    member_name: str = "hotpotqa/queries.jsonl",
) -> Iterator[QueryRecord]:
    """Stream query records incrementally from an archive.

    Args:
        source: Archive path or open ZipFile object.
        member_name: Archive path for queries JSONL.

    Yields:
        Validated QueryRecord instances.

    Raises:
        HotpotQAMissingMemberError: If member_name does not exist in the archive.
        HotpotQADataError: If any line contains invalid JSON or lacks required fields.
    """
    with _open_archive(source) as zf:
        if member_name not in zf.namelist():
            raise HotpotQAMissingMemberError(f"Missing archive member: {member_name}")
        with zf.open(member_name, "r") as raw_file:
            for line_no, raw_line in enumerate(raw_file, 1):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise HotpotQADataError(
                        f"Malformed JSON: {exc}",
                        file_name=member_name,
                        line_number=line_no,
                    ) from exc

                if not isinstance(item, dict):
                    raise HotpotQADataError(
                        "Record must be a JSON object",
                        file_name=member_name,
                        line_number=line_no,
                    )

                qid = str(item.get("_id", "")).strip()
                text = item.get("text")
                if not qid:
                    raise HotpotQADataError(
                        "Missing or empty '_id' field",
                        file_name=member_name,
                        line_number=line_no,
                    )
                if text is None or not isinstance(text, str):
                    raise HotpotQADataError(
                        "Missing or non-string 'text' field",
                        file_name=member_name,
                        line_number=line_no,
                    )

                metadata = item.get("metadata")
                meta_dict = metadata if isinstance(metadata, dict) else {}

                yield QueryRecord(id=qid, text=text, metadata=meta_dict)


def read_qrels(
    source: Path | str | zipfile.ZipFile,
    split: str = "dev",
    member_name: str | None = None,
) -> list[QrelRecord]:
    """Read relevance judgments for a specific split.

    Args:
        source: Archive path or open ZipFile object.
        split: Dataset split ('dev', 'test', or 'train').
        member_name: Optional override for the TSV path within the archive.

    Returns:
        List of QrelRecord instances.

    Raises:
        HotpotQAMissingMemberError: If the TSV file does not exist in the archive.
        HotpotQADataError: If header or data rows are malformed.
    """
    target_member = member_name or f"hotpotqa/qrels/{split}.tsv"
    expected_header = ["query-id", "corpus-id", "score"]

    with _open_archive(source) as zf:
        if target_member not in zf.namelist():
            raise HotpotQAMissingMemberError(f"Missing archive member: {target_member}")

        records: list[QrelRecord] = []
        with zf.open(target_member, "r") as raw_file:
            text_stream = io.TextIOWrapper(raw_file, encoding="utf-8")
            reader = csv.reader(text_stream, delimiter="\t")
            header = next(reader, None)
            if header is None:
                raise HotpotQADataError(
                    "Empty qrels file",
                    file_name=target_member,
                    line_number=1,
                )

            normalized_header = [h.strip() for h in header]
            if normalized_header != expected_header:
                raise HotpotQADataError(
                    f"Invalid header {header}, expected {expected_header}",
                    file_name=target_member,
                    line_number=1,
                )

            for line_no, row in enumerate(reader, 2):
                if not row or not any(row):
                    continue
                if len(row) != 3:
                    raise HotpotQADataError(
                        f"Expected 3 columns, got {len(row)}",
                        file_name=target_member,
                        line_number=line_no,
                    )

                qid = row[0].strip()
                cid = row[1].strip()
                score_str = row[2].strip()

                if not qid:
                    raise HotpotQADataError(
                        "Missing query-id",
                        file_name=target_member,
                        line_number=line_no,
                    )
                if not cid:
                    raise HotpotQADataError(
                        "Missing corpus-id",
                        file_name=target_member,
                        line_number=line_no,
                    )
                try:
                    score = int(score_str)
                except ValueError as exc:
                    raise HotpotQADataError(
                        f"Invalid integer score: {score_str}",
                        file_name=target_member,
                        line_number=line_no,
                    ) from exc

                records.append(QrelRecord(query_id=qid, corpus_id=cid, score=score))

        return records


def audit_hotpotqa_archive(archive_path: Path | str) -> dict[str, Any]:
    """Perform a factual structural audit of the HotpotQA archive.

    Streams files to avoid loading full corpus into memory.

    Args:
        archive_path: Path to hotpotqa.zip.

    Returns:
        Structured audit report mapping.

    Raises:
        FileNotFoundError: If the archive does not exist.
        HotpotQAError: If the archive is corrupt or invalid.
    """
    path = Path(archive_path)
    if not path.exists():
        raise FileNotFoundError(f"Archive file not found: {path}")

    file_size = path.stat().st_size
    sha256_hash = compute_sha256(path)

    if not zipfile.is_zipfile(path):
        raise HotpotQAError(f"Not a valid ZIP archive: {path}")

    with zipfile.ZipFile(path, "r") as zf:
        corrupt_entry = zf.testzip()
        if corrupt_entry is not None:
            raise HotpotQAError(f"Archive corrupted at entry: {corrupt_entry}")

        namelist = zf.namelist()
        required_members = [
            "hotpotqa/corpus.jsonl",
            "hotpotqa/queries.jsonl",
            "hotpotqa/qrels/dev.tsv",
            "hotpotqa/qrels/test.tsv",
        ]
        for req in required_members:
            if req not in namelist:
                raise HotpotQAMissingMemberError(
                    f"Archive missing required member: {req}"
                )

        contents = [
            {"path": info.filename, "size_bytes": info.file_size}
            for info in zf.infolist()
        ]

        # Audit queries
        query_ids: set[str] = set()
        duplicate_queries = 0
        for q in iter_queries(zf):
            if q.id in query_ids:
                duplicate_queries += 1
            query_ids.add(q.id)

        # Audit qrels
        qrel_splits: dict[str, list[QrelRecord]] = {
            "dev": read_qrels(zf, split="dev"),
            "test": read_qrels(zf, split="test"),
        }
        if "hotpotqa/qrels/train.tsv" in namelist:
            qrel_splits["train"] = read_qrels(zf, split="train")

        qrels_counts: dict[str, int] = {}
        all_qrel_qids: set[str] = set()
        all_qrel_cids: set[str] = set()

        for split_name, qrel_list in qrel_splits.items():
            qrels_counts[split_name] = len(qrel_list)
            for qrel in qrel_list:
                all_qrel_qids.add(qrel.query_id)
                all_qrel_cids.add(qrel.corpus_id)

        missing_qrel_queries = len(all_qrel_qids - query_ids)

        # Audit corpus incrementally
        corpus_count = 0
        found_qrel_cids: set[str] = set()
        for doc in iter_corpus(zf):
            corpus_count += 1
            if doc.id in all_qrel_cids:
                found_qrel_cids.add(doc.id)

        missing_qrel_passages = len(all_qrel_cids - found_qrel_cids)

        return {
            "file_size_bytes": file_size,
            "sha256": sha256_hash,
            "is_valid_zip": True,
            "archive_contents": contents,
            "record_counts": {
                "corpus": corpus_count,
                "queries": len(query_ids),
                "qrels": qrels_counts,
            },
            "data_quality": {
                "missing_or_malformed_records": 0,
                "duplicate_query_ids": duplicate_queries,
                "qrels_queries_missing_from_queries": missing_qrel_queries,
                "qrels_corpus_missing_from_corpus": missing_qrel_passages,
            },
            "audit_status": "verified",
        }
