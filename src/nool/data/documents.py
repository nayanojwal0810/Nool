"""Read company documents and store traceable, versioned passages."""

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
WORD_PATTERN = re.compile(r"\S+")


def source_digest(path: Path) -> str:
    """Hash text consistently across Windows and Unix line endings."""
    content = path.read_bytes()
    if path.suffix.lower() in {".md", ".txt"}:
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True)
class Section:
    name: str
    text: str
    page_number: int | None = None


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    title: str
    source_path: str
    owner: str
    version: int
    content_hash: str
    allowed_roles: tuple[str, ...]
    active: bool


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    version: int
    content_hash: str
    section: str
    page_number: int | None
    position: int
    text: str
    word_count: int


def read_sections(path: Path | str) -> list[Section]:
    """Extract text from Markdown, TXT, or a PDF with selectable text."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported document type: {suffix}")
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.stat().st_size > 5_000_000:
        raise ValueError(f"Document exceeds the 5 MB limit: {source}")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF reading requires the pypdf dependency") from exc
        reader = PdfReader(str(source))
        sections = [
            Section(f"Page {number}", page.extract_text() or "", number)
            for number, page in enumerate(reader.pages, start=1)
        ]
    else:
        content = source.read_text(encoding="utf-8-sig")
        if "\x00" in content:
            raise ValueError(f"Document contains binary data: {source}")
        if suffix == ".txt":
            sections = [Section("Body", content)]
        else:
            sections = []
            name = "Introduction"
            lines: list[str] = []
            for line in content.splitlines():
                if line.startswith("## "):
                    if "\n".join(lines).strip():
                        sections.append(Section(name, "\n".join(lines)))
                    name = line[3:].strip()
                    lines = []
                else:
                    lines.append(line)
            if "\n".join(lines).strip():
                sections.append(Section(name, "\n".join(lines)))

    sections = [s for s in sections if s.text.strip()]
    if not sections:
        raise ValueError(f"Document has no extractable text: {source}")
    return sections


def make_chunks(
    document_id: str,
    version: int,
    content_hash: str,
    sections: list[Section],
    *,
    words_per_chunk: int = 180,
    overlap: int = 30,
) -> list[ChunkRecord]:
    """Split sections into stable word windows while keeping source locations."""
    if words_per_chunk <= 0 or overlap < 0 or overlap >= words_per_chunk:
        raise ValueError("Chunk size must be positive and overlap smaller than size")
    chunks = []
    for section in sections:
        matches = list(WORD_PATTERN.finditer(section.text))
        step = words_per_chunk - overlap
        for start in range(0, len(matches), step):
            window = matches[start : start + words_per_chunk]
            if not window:
                break
            position = len(chunks)
            text = section.text[window[0].start() : window[-1].end()].strip()
            identity = (
                f"{document_id}:{version}:{content_hash}:"
                f"{section.page_number}:{section.name}:{position}"
            )
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            chunks.append(
                ChunkRecord(
                    chunk_id,
                    document_id,
                    version,
                    content_hash,
                    section.name,
                    section.page_number,
                    position,
                    text,
                    len(window),
                )
            )
            if start + words_per_chunk >= len(matches):
                break
    return chunks


class DocumentStore:
    """SQLite source of truth for current documents and their passage history."""

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    source_path TEXT NOT NULL, owner TEXT NOT NULL,
                    version INTEGER NOT NULL, content_hash TEXT NOT NULL,
                    allowed_roles TEXT NOT NULL, active INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
                    version INTEGER NOT NULL, content_hash TEXT NOT NULL,
                    section TEXT NOT NULL, page_number INTEGER,
                    position INTEGER NOT NULL, text TEXT NOT NULL,
                    word_count INTEGER NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(document_id)
                );
                CREATE INDEX IF NOT EXISTS chunks_by_document
                    ON chunks(document_id, version);
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def ingest(
        self,
        source_path: Path | str,
        *,
        document_id: str,
        title: str,
        owner: str,
        allowed_roles: list[str] | tuple[str, ...],
    ) -> DocumentRecord:
        """Create a document or advance its version when its content changes."""
        if not document_id.strip() or not title.strip() or not owner.strip():
            raise ValueError("Document ID, title, and owner are required")
        roles = tuple(sorted(set(allowed_roles)))
        if not roles or any(not role.strip() for role in roles):
            raise ValueError("At least one non-empty allowed role is required")
        path = Path(source_path).resolve()
        sections = read_sections(path)
        digest = source_digest(path)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT version, content_hash, allowed_roles "
                "FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            version = (existing["version"] if existing else 0) + (
                0 if existing and existing["content_hash"] == digest else 1
            )
            if existing and existing["allowed_roles"] != json.dumps(roles):
                raise ValueError("Change access through set_allowed_roles")
            chunks = make_chunks(document_id, version, digest, sections)
            connection.execute(
                """INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
                   ON CONFLICT(document_id) DO UPDATE SET
                   title=excluded.title, source_path=excluded.source_path,
                   owner=excluded.owner, version=excluded.version,
                   content_hash=excluded.content_hash,
                   updated_at=excluded.updated_at""",
                (
                    document_id,
                    title,
                    str(path),
                    owner,
                    version,
                    digest,
                    json.dumps(roles),
                    datetime.now(UTC).isoformat(),
                ),
            )
            connection.executemany(
                """INSERT OR IGNORE INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        c.chunk_id,
                        c.document_id,
                        c.version,
                        c.content_hash,
                        c.section,
                        c.page_number,
                        c.position,
                        c.text,
                        c.word_count,
                    )
                    for c in chunks
                ],
            )
        return self.get_document(document_id)

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        if row is None:
            return None
        return DocumentRecord(
            row["document_id"],
            row["title"],
            row["source_path"],
            row["owner"],
            row["version"],
            row["content_hash"],
            tuple(json.loads(row["allowed_roles"])),
            bool(row["active"]),
        )

    def list_documents(self) -> list[DocumentRecord]:
        with self._connect() as connection:
            ids = [
                row["document_id"]
                for row in connection.execute(
                    "SELECT document_id FROM documents ORDER BY document_id"
                )
            ]
        return [self.get_document(document_id) for document_id in ids]

    def current_chunks(self, document_id: str) -> list[ChunkRecord]:
        """Return passages from the current active version only."""
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT c.* FROM chunks c JOIN documents d
                   ON c.document_id = d.document_id AND c.version = d.version
                   WHERE d.document_id = ? AND d.active = 1
                   ORDER BY c.position""",
                (document_id,),
            ).fetchall()
        return [ChunkRecord(**dict(row)) for row in rows]

    def set_allowed_roles(self, document_id: str, roles: list[str]) -> None:
        allowed = tuple(sorted(set(roles)))
        if not allowed or any(not role.strip() for role in allowed):
            raise ValueError("At least one non-empty allowed role is required")
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE documents SET allowed_roles = ?, updated_at = ? "
                "WHERE document_id = ?",
                (json.dumps(allowed), datetime.now(UTC).isoformat(), document_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(document_id)

    def archive(self, document_id: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE documents SET active = 0, updated_at = ? WHERE document_id = ?",
                (datetime.now(UTC).isoformat(), document_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(document_id)

    def activate(self, document_id: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE documents SET active = 1, updated_at = ? WHERE document_id = ?",
                (datetime.now(UTC).isoformat(), document_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(document_id)
