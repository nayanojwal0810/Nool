"""BM25 sparse lexical retrieval implementation using bm25s."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import bm25s

INDEX_SCHEMA_VERSION = "1.0.0"
REQUIRED_INDEX_FILES = [
    "params.index.json",
    "vocab.index.json",
    "data.csc.index.npy",
    "indices.csc.index.npy",
    "indptr.csc.index.npy",
    "documents.jsonl",
    "index_manifest.json",
]


@dataclass(frozen=True)
class BM25SearchResult:
    """A single ranked search result from BM25 retrieval."""

    document_id: str
    title: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _IndexedDocument:
    """Internal representation of an indexed document aligned by position."""

    id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


class _SimpleEnglishStemmer:
    """Lightweight English suffix stemmer used when PyStemmer is not installed."""

    def stemWords(self, words: list[str]) -> list[str]:
        """Stem a list of words."""
        return [self.stem_word(w) for w in words]

    @staticmethod
    def stem_word(word: str) -> str:
        """Stem an individual English word using common suffix rules."""
        w = word.lower()
        if len(w) <= 3:
            return w
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith("tion") and len(w) > 5:
            return w[:-4] + "t"
        if w.endswith("ing") and len(w) > 4:
            base = w[:-3]
            if len(base) > 2 and base[-1] == base[-2] and base[-1] not in "lsz":
                return base[:-1]
            return base
        if w.endswith("ed") and len(w) > 3:
            base = w[:-2]
            if len(base) > 2 and base[-1] == base[-2] and base[-1] not in "lsz":
                return base[:-1]
            return base
        if w.endswith("es") and len(w) > 3:
            return w[:-2]
        if w.endswith("s") and not w.endswith("ss") and len(w) > 2:
            return w[:-1]
        return w


def resolve_stemmer(stemmer: Any) -> tuple[Any, str | None]:
    """Resolve a stemmer setting into an executable stemmer and identifier."""
    if stemmer is None or stemmer is False:
        return None, None

    if isinstance(stemmer, str):
        stemmer_name = stemmer.lower().strip()
        if stemmer_name in ("none", "null", "disabled", ""):
            return None, None
        if stemmer_name == "english":
            try:
                import Stemmer

                return Stemmer.Stemmer("english"), "english"
            except ImportError:
                return _SimpleEnglishStemmer(), "english"
        raise ValueError(f"Unsupported stemmer identifier: '{stemmer}'")

    if hasattr(stemmer, "stemWords") or callable(stemmer):
        identifier = getattr(stemmer, "name", None) or getattr(
            stemmer, "__name__", "custom"
        )
        return stemmer, str(identifier)

    raise ValueError(f"Invalid stemmer type: {type(stemmer)}")


class BM25Retriever:
    """BM25 sparse retriever for index building, loading, and searching."""

    def __init__(
        self,
        bm25: bm25s.BM25,
        documents: list[_IndexedDocument],
        manifest: dict[str, Any],
        stemmer: Any = None,
    ) -> None:
        """Initialize retriever with loaded BM25 model, document store, and manifest.

        Args:
            bm25: Pre-indexed or loaded bm25s.BM25 model.
            documents: List of indexed documents in exact position alignment.
            manifest: Index metadata dictionary.
            stemmer: Restored stemmer object or callable matching index tokenization.
        """
        self.bm25 = bm25
        self.documents = documents
        self.manifest = manifest
        self.stopwords = manifest.get("stopwords", "english")
        self.stemmer = stemmer
        self.stemmer_id = manifest.get("stemmer")

    @classmethod
    def build(
        cls,
        corpus_path: Path | str,
        index_dir: Path | str,
        method: str = "lucene",
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: str | list[str] | None = "english",
        stemmer: Any = None,
        indexed_fields: Sequence[str] = ("title", "text"),
        overwrite: bool = False,
    ) -> "BM25Retriever":
        """Build and persist a BM25 index from a JSONL corpus file.

        Args:
            corpus_path: Path to the source corpus JSONL file.
            index_dir: Target directory to save the index and document store.
            method: BM25 variant scoring algorithm (default: 'lucene').
            k1: Term frequency saturation parameter (default: 1.5).
            b: Document length normalization parameter (default: 0.75).
            stopwords: Stopwords configuration for tokenization (default: 'english').
            stemmer: Optional stemmer (default: None, stemming disabled).
            indexed_fields: Document fields to include in searchable text.
            overwrite: Whether to overwrite an existing index directory.

        Returns:
            Configured and loaded BM25Retriever instance.

        Raises:
            FileNotFoundError: If corpus_path does not exist.
            FileExistsError: If index_dir exists and overwrite is False.
            ValueError: If corpus is empty or contains malformed records.
        """
        corpus_file = Path(corpus_path)
        out_dir = Path(index_dir)

        if not corpus_file.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

        if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Index directory already exists and is not empty: {out_dir}. "
                "Set overwrite=True to rebuild."
            )

        documents: list[_IndexedDocument] = []
        searchable_texts: list[str] = []
        seen_ids: set[str] = set()

        with open(corpus_file, encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, 1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON in corpus at {corpus_file}:{line_no}: {exc}"
                    ) from exc

                if not isinstance(record, dict):
                    raise ValueError(
                        f"Record at {corpus_file}:{line_no} must be a JSON object."
                    )

                doc_id = str(record.get("_id", "")).strip()
                if not doc_id:
                    raise ValueError(
                        f"Missing or empty '_id' field at {corpus_file}:{line_no}."
                    )

                if doc_id in seen_ids:
                    raise ValueError(
                        f"Duplicate document ID '{doc_id}' encountered at "
                        f"{corpus_file}:{line_no}."
                    )
                seen_ids.add(doc_id)

                text = record.get("text")
                if text is None or not isinstance(text, str):
                    raise ValueError(
                        f"Missing or invalid 'text' field at {corpus_file}:{line_no}."
                    )

                title_raw = record.get("title")
                title = str(title_raw).strip() if title_raw is not None else ""
                metadata = record.get("metadata")
                meta_dict = metadata if isinstance(metadata, dict) else {}

                parts = []
                for field_name in indexed_fields:
                    if field_name == "title" and title:
                        parts.append(title)
                    elif field_name == "text" and text:
                        parts.append(text)

                combined_text = " ".join(parts).strip()
                if not combined_text:
                    combined_text = text.strip()

                documents.append(
                    _IndexedDocument(
                        id=doc_id,
                        title=title,
                        text=text,
                        metadata=meta_dict,
                    )
                )
                searchable_texts.append(combined_text)

        if not documents:
            raise ValueError(f"Corpus file {corpus_file} contains no valid records.")

        corpus_sha256 = _compute_sha256(corpus_file)
        stemmer_obj, stemmer_id = resolve_stemmer(stemmer)

        # 1. Lower-level tokenization
        corpus_tokens = bm25s.tokenize(
            searchable_texts,
            stopwords=stopwords,
            stemmer=stemmer_obj,
        )

        # 2. Lower-level BM25 indexing
        bm25_model = bm25s.BM25(method=method, k1=k1, b=b)
        bm25_model.index(corpus_tokens)

        # 3. Create index destination directory
        out_dir.mkdir(parents=True, exist_ok=True)

        # 4. Save BM25S index
        bm25_model.save(str(out_dir))

        # 5. Save aligned documents in matching index order
        docs_file = out_dir / "documents.jsonl"
        with open(docs_file, "w", encoding="utf-8", newline="\n") as f:
            for doc in documents:
                row = {
                    "_id": doc.id,
                    "title": doc.title,
                    "text": doc.text,
                }
                if doc.metadata:
                    row["metadata"] = doc.metadata
                f.write(json.dumps(row) + "\n")

        # 6. Save index manifest
        manifest = {
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "bm25s_library": "bm25s",
            "bm25s_version": getattr(bm25s, "__version__", "0.3.16"),
            "method": method,
            "k1": k1,
            "b": b,
            "stopwords": stopwords,
            "stemming": stemmer_id is not None,
            "stemmer": stemmer_id,
            "indexed_fields": list(indexed_fields),
            "document_count": len(documents),
            "source_corpus_filename": corpus_file.name,
            "source_corpus_sha256": corpus_sha256,
            "created_at": datetime.now(UTC).isoformat(),
        }

        manifest_file = out_dir / "index_manifest.json"
        with open(manifest_file, "w", encoding="utf-8", newline="\n") as f:
            json.dump(manifest, f, indent=2)

        return cls(
            bm25=bm25_model,
            documents=documents,
            manifest=manifest,
            stemmer=stemmer_obj,
        )

    @classmethod
    def load(cls, index_dir: Path | str) -> "BM25Retriever":
        """Load a persisted BM25S index, aligned documents, and manifest.

        Args:
            index_dir: Directory containing saved index files.

        Returns:
            Initialized BM25Retriever instance ready for searching.

        Raises:
            FileNotFoundError: If index directory or required files are missing.
            ValueError: If index files or manifest are corrupt.
        """
        dir_path = Path(index_dir)
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Index directory not found: {dir_path}")

        missing_files = [f for f in REQUIRED_INDEX_FILES if not (dir_path / f).exists()]
        if missing_files:
            missing_str = ", ".join(missing_files)
            raise FileNotFoundError(
                f"Incomplete BM25 index in {dir_path}. Missing: {missing_str}"
            )

        manifest_path = dir_path / "index_manifest.json"
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt index manifest {manifest_path}: {exc}") from exc

        # Load BM25S model
        bm25_model = bm25s.BM25.load(str(dir_path), load_corpus=False)

        # Load aligned documents
        docs_path = dir_path / "documents.jsonl"
        documents: list[_IndexedDocument] = []
        with open(docs_path, encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, 1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Corrupt document record at {docs_path}:{line_no}: {exc}"
                    ) from exc

                documents.append(
                    _IndexedDocument(
                        id=record["_id"],
                        title=record.get("title", ""),
                        text=record["text"],
                        metadata=record.get("metadata", {}),
                    )
                )

        expected_count = manifest.get("document_count")
        if expected_count is not None and len(documents) != expected_count:
            raise ValueError(
                f"Document count mismatch in {dir_path}: manifest expects "
                f"{expected_count}, but documents.jsonl has {len(documents)}."
            )

        saved_stemmer_id = manifest.get("stemmer")
        stemmer_obj, _ = resolve_stemmer(saved_stemmer_id)

        return cls(
            bm25=bm25_model,
            documents=documents,
            manifest=manifest,
            stemmer=stemmer_obj,
        )

    def search(self, query: str, top_k: int = 10) -> list[BM25SearchResult]:
        """Search the indexed passages using BM25 scoring.

        Args:
            query: Plaintext question or query string.
            top_k: Number of highest-ranked passages to return (must be positive).

        Returns:
            List of BM25SearchResult sorted by score descending (1-based ranks).
            Returns empty list if query has no matching terms or positive scores.

        Raises:
            ValueError: If query is blank or top_k is not a positive integer.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query string cannot be empty or blank.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got: {top_k}")

        if not self.documents:
            return []

        query_tokens = bm25s.tokenize(
            query,
            stopwords=self.stopwords,
            stemmer=self.stemmer,
        )

        # If query contains only stopwords or unknown tokens with no IDs
        token_ids = getattr(query_tokens, "ids", None)
        if token_ids is not None and (len(token_ids) == 0 or len(token_ids[0]) == 0):
            return []

        k = min(top_k, len(self.documents))
        doc_indices, scores = self.bm25.retrieve(
            query_tokens,
            k=k,
            show_progress=False,
        )

        if len(doc_indices) == 0 or len(scores) == 0:
            return []

        q_indices = doc_indices[0]
        q_scores = scores[0]

        results: list[BM25SearchResult] = []
        current_rank = 1

        for idx, score in zip(q_indices, q_scores, strict=True):
            float_score = float(score)
            if float_score <= 0.0:
                continue

            doc = self.documents[int(idx)]
            results.append(
                BM25SearchResult(
                    document_id=doc.id,
                    title=doc.title,
                    text=doc.text,
                    score=float_score,
                    rank=current_rank,
                    metadata=doc.metadata,
                )
            )
            current_rank += 1

        return results
