"""Contriever dense retrieval implementation using exact dot-product similarity."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODEL_ID = "facebook/contriever"
DEFAULT_MODEL_REVISION = "2bd46a25019aeea091fd42d1f0fd4801675cf699"
DEFAULT_EMBEDDING_DIM = 768
DEFAULT_MAX_SEQ_LENGTH = 512
BENCHMARK_CORPUS_SHA256 = (
    "c3970d91ebfd36d794066b42bd7acb85a744aabba2c97dd9a9bfde52d3dcbfbf"
)
BENCHMARK_PASSAGE_COUNT = 50000
ARTIFACT_SCHEMA_VERSION = "1.0.0"
REQUIRED_ARTIFACT_FILES = [
    "passage_embeddings.npy",
    "document_ids.json",
    "index_manifest.json",
]


@dataclass(frozen=True)
class DenseSearchResult:
    """A single ranked search result from Contriever dense retrieval."""

    document_id: str
    title: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_sha256(file_path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_corpus_file(
    corpus_path: Path | str,
    expected_sha256: str | None = None,
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
    """Validate source corpus JSONL file integrity, types, and schema.

    Args:
        corpus_path: Path to source corpus JSONL file.
        expected_sha256: Optional expected SHA-256 checksum to enforce.
        expected_count: Optional expected passage count to enforce.

    Returns:
        List of parsed, validated document dictionaries in exact corpus order.

    Raises:
        FileNotFoundError: If corpus_path does not exist.
        ValueError: If any validation rule fails.
    """
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Source corpus file not found: {path}")

    actual_sha256 = compute_sha256(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"Source corpus SHA-256 mismatch for {path}:\n"
            f"  Expected: {expected_sha256}\n"
            f"  Actual:   {actual_sha256}"
        )

    records: list[dict[str, Any]] = []
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
                raise ValueError(
                    f"Corpus record at {path}:{line_no} must be a JSON object, "
                    f"got {type(record).__name__}."
                )

            # Validate _id
            if "_id" not in record:
                raise ValueError(f"Missing '_id' in corpus record at {path}:{line_no}.")
            doc_id = record["_id"]
            if not isinstance(doc_id, str):
                raise ValueError(
                    f"Field '_id' at {path}:{line_no} must be a string, "
                    f"got {type(doc_id).__name__}."
                )
            if not doc_id.strip():
                raise ValueError(
                    f"Field '_id' at {path}:{line_no} cannot be empty or whitespace."
                )

            if doc_id in seen_ids:
                raise ValueError(
                    f"Duplicate document ID found in corpus at {path}:{line_no}: "
                    f"'{doc_id}'."
                )
            seen_ids.add(doc_id)

            # Validate text
            if "text" not in record:
                raise ValueError(
                    f"Missing 'text' in corpus record at {path}:{line_no}."
                )
            text_val = record["text"]
            if not isinstance(text_val, str):
                raise ValueError(
                    f"Field 'text' at {path}:{line_no} must be a string, "
                    f"got {type(text_val).__name__}."
                )

            # Validate title if present (must be a string; null is rejected)
            if "title" in record:
                title_val = record["title"]
                if not isinstance(title_val, str):
                    raise ValueError(
                        f"Field 'title' at {path}:{line_no} must be a string, "
                        f"got {type(title_val).__name__}."
                    )

            records.append(record)

    if expected_count is not None and len(records) != expected_count:
        raise ValueError(
            f"Corpus passage count mismatch for {path}: "
            f"got {len(records)}, expected exactly {expected_count}."
        )

    return records


def validate_checkpoint_shard(
    shard_path: Path | str,
    expected_start: int,
    expected_end: int,
    expected_doc_ids: list[str],
    expected_corpus_sha256: str,
    expected_model_id: str = DEFAULT_MODEL_ID,
    expected_model_revision: str = DEFAULT_MODEL_REVISION,
    expected_dim: int = DEFAULT_EMBEDDING_DIM,
    expected_max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
) -> None:
    """Validate that a checkpoint shard is complete, uncorrupted, and matches metadata.

    Args:
        shard_path: Path to shard .npz archive.
        expected_start: Expected 0-based start index.
        expected_end: Expected 0-based end index (exclusive).
        expected_doc_ids: Expected document IDs in exact sequence.
        expected_corpus_sha256: Expected SHA-256 of the source corpus.
        expected_model_id: Locked model identifier.
        expected_model_revision: Locked model revision SHA.
        expected_dim: Locked embedding dimension (768).
        expected_max_seq_length: Locked sequence length (512).

    Raises:
        FileNotFoundError: If shard file is missing.
        ValueError: If any validation criterion fails.
    """
    path = Path(shard_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint shard file not found: {path}")

    expected_count = len(expected_doc_ids)
    if expected_count != (expected_end - expected_start):
        raise ValueError(
            f"Inconsistent expected indices: start={expected_start}, "
            f"end={expected_end}, but expected_doc_ids has {expected_count} items."
        )

    with np.load(str(path), allow_pickle=False) as data:
        # 1. Required keys
        required_keys = {
            "embeddings",
            "doc_ids",
            "start_idx",
            "end_idx",
            "source_corpus_sha256",
            "model_id",
            "model_revision",
            "pooling",
            "max_seq_length",
            "output_dtype",
            "batch_sizes_used",
        }
        missing_keys = required_keys - set(data.files)
        if missing_keys:
            raise ValueError(
                f"Checkpoint shard {path.name} is missing required fields: "
                f"{sorted(missing_keys)}."
            )

        # 2. Start and end indices
        start_idx = int(data["start_idx"])
        end_idx = int(data["end_idx"])
        if start_idx != expected_start or end_idx != expected_end:
            raise ValueError(
                f"Shard {path.name} index mismatch: expected "
                f"({expected_start}, {expected_end}), got ({start_idx}, {end_idx})."
            )

        # 3. Document IDs dtype and exact match
        doc_ids_arr = data["doc_ids"]
        if doc_ids_arr.dtype.kind not in ("U", "S"):
            raise ValueError(
                f"Shard {path.name} document IDs must be a NumPy string array, "
                f"got dtype {doc_ids_arr.dtype}."
            )

        shard_doc_ids = [str(x) for x in doc_ids_arr]
        if len(shard_doc_ids) != expected_count:
            raise ValueError(
                f"Shard {path.name} ID count mismatch: expected {expected_count}, "
                f"got {len(shard_doc_ids)}."
            )
        if shard_doc_ids != expected_doc_ids:
            for i, (actual_id, exp_id) in enumerate(
                zip(shard_doc_ids, expected_doc_ids, strict=False)
            ):
                if actual_id != exp_id:
                    raise ValueError(
                        f"Shard {path.name} ID mismatch at offset {i}: "
                        f"expected '{exp_id}', got '{actual_id}'."
                    )

        # 4. Source corpus SHA-256
        corpus_sha = str(data["source_corpus_sha256"])
        if corpus_sha != expected_corpus_sha256:
            raise ValueError(
                f"Shard {path.name} corpus SHA-256 mismatch: expected "
                f"{expected_corpus_sha256}, got {corpus_sha}."
            )

        # 5. Model ID & revision
        model_id = str(data["model_id"])
        if model_id != expected_model_id:
            raise ValueError(
                f"Shard {path.name} model ID mismatch: expected "
                f"'{expected_model_id}', got '{model_id}'."
            )

        model_rev = str(data["model_revision"])
        if model_rev != expected_model_revision:
            raise ValueError(
                f"Shard {path.name} model revision mismatch: expected "
                f"'{expected_model_revision}', got '{model_rev}'."
            )

        # 6. Pooling method
        pooling = str(data["pooling"])
        if pooling != "masked_mean":
            raise ValueError(
                f"Shard {path.name} pooling mismatch: expected 'masked_mean', "
                f"got '{pooling}'."
            )

        # 7. Maximum sequence length
        max_len = int(data["max_seq_length"])
        if max_len != expected_max_seq_length:
            raise ValueError(
                f"Shard {path.name} max sequence length mismatch: expected "
                f"{expected_max_seq_length}, got {max_len}."
            )

        # 8. Output dtype
        out_dtype = str(data["output_dtype"])
        if out_dtype != "float32":
            raise ValueError(
                f"Shard {path.name} output dtype mismatch: expected 'float32', "
                f"got '{out_dtype}'."
            )

        # 9. Embedding matrix shape, dimension, dtype, and finiteness
        embeddings = data["embeddings"]
        if embeddings.ndim != 2:
            raise ValueError(
                f"Shard {path.name} embeddings must be 2D, got {embeddings.ndim}D."
            )
        if embeddings.shape != (expected_count, expected_dim):
            raise ValueError(
                f"Shard {path.name} embedding shape mismatch: expected "
                f"({expected_count}, {expected_dim}), got {embeddings.shape}."
            )
        if embeddings.dtype != np.float32:
            raise ValueError(
                f"Shard {path.name} embedding dtype mismatch: expected float32, "
                f"got {embeddings.dtype}."
            )
        if not np.isfinite(embeddings).all():
            raise ValueError(
                f"Shard {path.name} embeddings contain non-finite (NaN or Inf) values."
            )

        # 10. Batch sizes used metadata
        batch_sizes_arr = data["batch_sizes_used"]
        if batch_sizes_arr.dtype.kind not in ("i", "u"):
            raise ValueError(
                f"Shard {path.name} batch_sizes_used must be an integer array, "
                f"got dtype {batch_sizes_arr.dtype}."
            )
        if batch_sizes_arr.ndim != 1:
            raise ValueError(
                f"Shard {path.name} batch_sizes_used must be 1D, "
                f"got {batch_sizes_arr.ndim}D."
            )
        if batch_sizes_arr.size == 0:
            raise ValueError(f"Shard {path.name} batch_sizes_used cannot be empty.")
        if (batch_sizes_arr <= 0).any():
            raise ValueError(
                f"Shard {path.name} batch_sizes_used must contain only positive "
                f"integers, got {batch_sizes_arr.tolist()}."
            )
        if len(np.unique(batch_sizes_arr)) != len(batch_sizes_arr):
            raise ValueError(
                f"Shard {path.name} batch_sizes_used values must be unique, "
                f"got {batch_sizes_arr.tolist()}."
            )


def verify_artifact(
    artifact_dir: Path | str,
    corpus_path: Path | str,
    expected_passage_count: int | None = None,
    expected_corpus_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate Contriever artifact files, manifest, checksums, and alignment.

    Args:
        artifact_dir: Path to directory containing artifact files.
        corpus_path: Path to source corpus JSONL file.
        expected_passage_count: Optional expected passage count to enforce.
        expected_corpus_sha256: Optional expected corpus SHA-256 to enforce.

    Returns:
        Summary metrics dictionary.

    Raises:
        FileNotFoundError: If directory or required files are missing.
        ValueError: If validation checks fail.
    """
    art_dir = Path(artifact_dir)
    corp_path = Path(corpus_path)

    if not art_dir.exists() or not art_dir.is_dir():
        raise FileNotFoundError(f"Artifact directory not found: {art_dir}")

    if not corp_path.exists():
        raise FileNotFoundError(f"Source corpus not found: {corp_path}")

    # 1. Check required files
    missing = [f for f in REQUIRED_ARTIFACT_FILES if not (art_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required artifact file(s) in {art_dir}: {', '.join(missing)}"
        )

    # 2. Parse and validate manifest
    manifest_file = art_dir / "index_manifest.json"
    try:
        with open(manifest_file, encoding="utf-8") as f:
            manifest = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupt manifest JSON in {manifest_file}: {exc}") from exc

    if not isinstance(manifest, dict):
        raise ValueError("index_manifest.json must be a JSON object.")

    required_manifest_keys = [
        "model_id",
        "model_revision",
        "embedding_dimension",
        "pooling",
        "normalize_embeddings",
        "output_dtype",
        "source_corpus_sha256",
        "embedding_sha256",
        "document_ids_sha256",
        "passage_count",
    ]
    for key in required_manifest_keys:
        if key not in manifest:
            raise ValueError(f"Missing required manifest field: '{key}'.")

    if "similarity_method" not in manifest and "similarity" not in manifest:
        raise ValueError(
            "Missing required manifest field: 'similarity_method' or 'similarity'."
        )
    if "max_token_length" not in manifest and "max_seq_length" not in manifest:
        raise ValueError(
            "Missing required manifest field: 'max_token_length' or 'max_seq_length'."
        )

    # Validate model configuration
    if manifest["model_id"] != DEFAULT_MODEL_ID:
        raise ValueError(
            f"Manifest model_id is '{manifest['model_id']}', "
            f"expected '{DEFAULT_MODEL_ID}'."
        )
    if manifest["model_revision"] != DEFAULT_MODEL_REVISION:
        raise ValueError(
            f"Manifest model_revision is '{manifest['model_revision']}', "
            f"expected '{DEFAULT_MODEL_REVISION}'."
        )

    expected_dim = int(manifest["embedding_dimension"])
    if expected_dim != DEFAULT_EMBEDDING_DIM:
        raise ValueError(
            f"Manifest embedding dimension is {expected_dim}, "
            f"expected {DEFAULT_EMBEDDING_DIM}."
        )

    if manifest["pooling"] != "masked_mean":
        raise ValueError(
            f"Manifest pooling is '{manifest['pooling']}', expected 'masked_mean'."
        )

    max_seq_len = int(
        manifest.get("max_token_length", manifest.get("max_seq_length", 0))
    )
    if max_seq_len != DEFAULT_MAX_SEQ_LENGTH:
        raise ValueError(
            f"Manifest sequence length is {max_seq_len}, "
            f"expected {DEFAULT_MAX_SEQ_LENGTH}."
        )

    if manifest["normalize_embeddings"] is not False:
        raise ValueError(
            f"Manifest normalize_embeddings is {manifest['normalize_embeddings']}, "
            "expected False."
        )

    similarity = manifest.get("similarity_method", manifest.get("similarity", ""))
    if similarity != "dot_product":
        raise ValueError(
            f"Manifest similarity method is '{similarity}', expected 'dot_product'."
        )

    if manifest["output_dtype"] != "float32":
        raise ValueError(
            f"Manifest output_dtype is '{manifest['output_dtype']}', "
            "expected 'float32'."
        )

    manifest_passage_count = int(manifest["passage_count"])
    if (
        expected_passage_count is not None
        and manifest_passage_count != expected_passage_count
    ):
        raise ValueError(
            f"Manifest passage count is {manifest_passage_count}, "
            f"expected {expected_passage_count}."
        )

    manifest_corpus_sha = str(manifest["source_corpus_sha256"])
    if (
        expected_corpus_sha256 is not None
        and manifest_corpus_sha != expected_corpus_sha256
    ):
        raise ValueError(
            f"Manifest source corpus SHA-256 is '{manifest_corpus_sha}', "
            f"expected '{expected_corpus_sha256}'."
        )

    # 3. Checksum validations
    corpus_sha256 = compute_sha256(corp_path)
    if corpus_sha256 != manifest_corpus_sha:
        raise ValueError(
            f"Source corpus SHA-256 mismatch!\n"
            f"  Calculated: {corpus_sha256}\n"
            f"  Manifest:   {manifest_corpus_sha}"
        )

    emb_file = art_dir / "passage_embeddings.npy"
    emb_sha256 = compute_sha256(emb_file)
    if emb_sha256 != manifest["embedding_sha256"]:
        raise ValueError(
            f"Passage embeddings SHA-256 mismatch!\n"
            f"  Calculated: {emb_sha256}\n"
            f"  Manifest:   {manifest['embedding_sha256']}"
        )

    doc_ids_file = art_dir / "document_ids.json"
    doc_ids_sha256 = compute_sha256(doc_ids_file)
    if doc_ids_sha256 != manifest["document_ids_sha256"]:
        raise ValueError(
            f"Document IDs SHA-256 mismatch!\n"
            f"  Calculated: {doc_ids_sha256}\n"
            f"  Manifest:   {manifest['document_ids_sha256']}"
        )

    # 4. Load document IDs and validate
    try:
        with open(doc_ids_file, encoding="utf-8") as f:
            document_ids = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupt document_ids.json in {doc_ids_file}: {exc}") from exc

    if not isinstance(document_ids, list):
        raise ValueError("document_ids.json must be a JSON list.")

    if len(document_ids) != manifest_passage_count:
        raise ValueError(
            f"Document ID count mismatch: got {len(document_ids)}, "
            f"expected {manifest_passage_count} from manifest."
        )

    seen_ids: set[str] = set()
    for idx, doc_id in enumerate(document_ids):
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError(f"Invalid or empty document ID at index {idx}.")
        if doc_id in seen_ids:
            raise ValueError(f"Duplicate document ID found: '{doc_id}'.")
        seen_ids.add(doc_id)

    # 5. Validate source corpus strictly and verify exact 1:1 order alignment
    corpus_records = validate_corpus_file(
        corp_path,
        expected_sha256=manifest_corpus_sha,
        expected_count=manifest_passage_count,
    )
    corpus_ids = [r["_id"] for r in corpus_records]

    if corpus_ids != document_ids:
        for idx, (cid, did) in enumerate(zip(corpus_ids, document_ids, strict=False)):
            if cid != did:
                raise ValueError(
                    f"Document ID alignment mismatch at index {idx}: "
                    f"corpus has '{cid}', artifact has '{did}'."
                )
        raise ValueError("Document IDs do not match corpus IDs exactly in order.")

    # 6. Load embeddings with memory mapping and validate
    embeddings = np.load(str(emb_file), mmap_mode="r")

    if embeddings.ndim != 2:
        raise ValueError(f"Embedding array must be 2D, got shape {embeddings.shape}.")

    if embeddings.shape[0] != len(document_ids):
        raise ValueError(
            f"Embedding row count ({embeddings.shape[0]}) does not match "
            f"document IDs count ({len(document_ids)})."
        )

    if embeddings.shape[1] != DEFAULT_EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension ({embeddings.shape[1]}) does not match "
            f"expected dimension ({DEFAULT_EMBEDDING_DIM})."
        )

    if embeddings.dtype != np.float32:
        raise ValueError(f"Embedding dtype must be float32, got {embeddings.dtype}.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings matrix contains NaN or Inf values.")

    return {
        "passage_count": len(document_ids),
        "embedding_dim": embeddings.shape[1],
        "dtype": str(embeddings.dtype),
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "pooling": manifest["pooling"],
        "similarity": similarity,
        "corpus_sha256": corpus_sha256,
        "embedding_sha256": emb_sha256,
        "document_ids_sha256": doc_ids_sha256,
    }


def mean_pooling(token_embeddings: Any, attention_mask: Any) -> Any:
    """Perform attention-mask-aware mean pooling matching Contriever convention.

    Supports both PyTorch tensors and NumPy ndarrays.

    Args:
        token_embeddings: Tensor or ndarray of shape (batch, seq_len, hidden_dim).
        attention_mask: Tensor or ndarray of shape (batch, seq_len) with 1 for valid.

    Returns:
        Pooled representation of shape (batch_size, hidden_dim).
    """
    if isinstance(token_embeddings, np.ndarray):
        expanded_mask = np.asarray(attention_mask)[..., np.newaxis]
        masked = np.where(expanded_mask.astype(bool), token_embeddings, 0.0)
        mask_sum = np.maximum(expanded_mask.sum(axis=1), 1e-9)
        return (masked.sum(axis=1) / mask_sum).astype(np.float32)

    # PyTorch tensor path
    token_embeddings = token_embeddings.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    mask_sum = attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
    return token_embeddings.sum(dim=1) / mask_sum


class ContrieverEncoder:
    """Reusable text encoder using the pretrained Contriever model."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        model_revision: str = DEFAULT_MODEL_REVISION,
        device: str = "auto",
        max_length: int = DEFAULT_MAX_SEQ_LENGTH,
    ) -> None:
        """Initialize ContrieverEncoder with model ID, revision, and device setting.

        Args:
            model_id: Hugging Face model identifier (default: facebook/contriever).
            model_revision: Immutable 40-character commit SHA.
            device: 'cpu', 'cuda', or 'auto'.
            max_length: Maximum tokenization sequence length.
        """
        self.model_id = model_id
        self.model_revision = model_revision
        self.device_setting = device
        self.max_length = max_length
        self._device: Any = None
        self._tokenizer: Any = None
        self._model: Any = None

    def _ensure_model_and_tokenizer(self) -> None:
        """Lazily load PyTorch, Transformers, model weights, and tokenizer."""
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Dense retrieval requires the 'dense' extra dependencies "
                "(torch and transformers). Please install them using:\n"
                'pip install -e ".[dense]"'
            ) from exc

        # Resolve compute device
        if self.device_setting == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.device_setting == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA device explicitly requested but CUDA is not available."
                )
            resolved = "cuda"
        elif self.device_setting == "cpu":
            resolved = "cpu"
        else:
            raise ValueError(
                f"Unsupported device setting: '{self.device_setting}'. "
                "Expected 'auto', 'cpu', or 'cuda'."
            )

        self._device = torch.device(resolved)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.model_revision,
        )
        self._model = AutoModel.from_pretrained(
            self.model_id,
            revision=self.model_revision,
        )
        self._model.eval()
        self._model.to(self._device)

        hidden_size = getattr(self._model.config, "hidden_size", None)
        if hidden_size != DEFAULT_EMBEDDING_DIM:
            raise ValueError(
                f"Contriever model hidden dimension is {hidden_size}, "
                f"expected {DEFAULT_EMBEDDING_DIM}."
            )

    def encode_texts(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Encode a list of texts into float32 embeddings.

        Args:
            texts: List of strings to encode.
            batch_size: Number of texts per forward pass.

        Returns:
            2D numpy array of shape (len(texts), 768) and dtype float32.
        """
        if not texts:
            return np.empty((0, DEFAULT_EMBEDDING_DIM), dtype=np.float32)

        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")

        self._ensure_model_and_tokenizer()
        import torch

        all_embeddings: list[np.ndarray] = []

        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx : start_idx + batch_size]
            inputs = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                pooled = mean_pooling(outputs[0], inputs["attention_mask"])

            batch_emb = pooled.detach().cpu().to(torch.float32).numpy()
            if not np.isfinite(batch_emb).all():
                raise ValueError("Model generated non-finite embedding values.")

            all_embeddings.append(batch_emb)

        result = np.vstack(all_embeddings)
        if result.shape[1] != DEFAULT_EMBEDDING_DIM:
            raise ValueError(
                f"Unexpected embedding dimension: got {result.shape[1]}, "
                f"expected {DEFAULT_EMBEDDING_DIM}."
            )
        return result.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single search query into a 1D float32 embedding vector.

        Args:
            query: Plaintext user question or query.

        Returns:
            1D numpy array of shape (768,) and dtype float32.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query string cannot be empty or blank.")

        emb_matrix = self.encode_texts([query.strip()], batch_size=1)
        return emb_matrix[0]


class ContrieverRetriever:
    """Dense retriever using precomputed Contriever passage embeddings."""

    def __init__(
        self,
        embeddings: np.ndarray,
        document_ids: list[str],
        corpus_documents: dict[str, dict[str, Any]],
        manifest: dict[str, Any],
        encoder: Any = None,
    ) -> None:
        """Initialize ContrieverRetriever.

        Args:
            embeddings: 2D numpy array of passage vectors of shape (N, 768).
            document_ids: Aligned list of document IDs in exact order.
            corpus_documents: Map from document_id to document dict.
            manifest: Index manifest dictionary.
            encoder: ContrieverEncoder or duck-typed encoder object.
        """
        if not isinstance(embeddings, np.ndarray):
            raise ValueError(
                f"embeddings must be a numpy ndarray, got {type(embeddings).__name__}"
            )
        if embeddings.ndim != 2:
            raise ValueError(
                f"Embedding matrix must be 2-dimensional, got {embeddings.ndim}D."
            )
        expected_dim = int(manifest.get("embedding_dimension", DEFAULT_EMBEDDING_DIM))
        if expected_dim != DEFAULT_EMBEDDING_DIM:
            raise ValueError(
                f"Manifest embedding dimension is {expected_dim}, "
                f"expected {DEFAULT_EMBEDDING_DIM}."
            )
        if embeddings.shape[1] != DEFAULT_EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: matrix has dimension "
                f"{embeddings.shape[1]}, expected {DEFAULT_EMBEDDING_DIM}."
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings matrix contains NaN or Inf values.")
        if embeddings.shape[0] != len(document_ids):
            raise ValueError(
                f"Count mismatch: embeddings has {embeddings.shape[0]} rows, "
                f"but document_ids has {len(document_ids)} items."
            )

        # Validate unique IDs
        seen_ids: set[str] = set()
        for doc_id in document_ids:
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValueError("Empty or invalid document ID found in document_ids.")
            if doc_id in seen_ids:
                raise ValueError(f"Duplicate document ID found: '{doc_id}'.")
            seen_ids.add(doc_id)
            if doc_id not in corpus_documents:
                raise ValueError(
                    f"Document ID '{doc_id}' not found in source corpus mapping."
                )

        self.embeddings = embeddings.astype(np.float32)
        self.document_ids = document_ids
        self.corpus_documents = corpus_documents
        self.manifest = manifest
        self.encoder = encoder

    @classmethod
    def load(
        cls,
        artifact_dir: Path | str,
        corpus_path: Path | str,
        encoder: Any = None,
        expected_passage_count: int | None = None,
        expected_corpus_sha256: str | None = None,
    ) -> "ContrieverRetriever":
        """Load artifact files, validate checksums, and initialize retriever.

        Performs full artifact verification using verify_artifact before loading.

        Args:
            artifact_dir: Path to directory containing Contriever artifact files.
            corpus_path: Path to source corpus JSONL file.
            encoder: Optional encoder instance to reuse.
            expected_passage_count: Optional expected passage count.
            expected_corpus_sha256: Optional expected corpus SHA-256.

        Returns:
            Configured ContrieverRetriever instance.
        """
        # Run shared verification logic - fails fast on any invalid metadata/data
        verify_artifact(
            artifact_dir=artifact_dir,
            corpus_path=corpus_path,
            expected_passage_count=expected_passage_count,
            expected_corpus_sha256=expected_corpus_sha256,
        )

        art_dir = Path(artifact_dir)
        corp_file = Path(corpus_path)

        with open(art_dir / "index_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)

        with open(art_dir / "document_ids.json", encoding="utf-8") as f:
            document_ids = json.load(f)

        corpus_records = validate_corpus_file(corp_file)
        corpus_docs = {r["_id"]: r for r in corpus_records}

        emb_file = art_dir / "passage_embeddings.npy"
        embeddings = np.load(str(emb_file), mmap_mode="r")

        return cls(
            embeddings=embeddings,
            document_ids=document_ids,
            corpus_documents=corpus_docs,
            manifest=manifest,
            encoder=encoder,
        )

    def search(self, query: str, top_k: int = 10) -> list[DenseSearchResult]:
        """Search passages using exact dot-product query similarity.

        Args:
            query: Question or query string.
            top_k: Number of highest-ranked results to return.

        Returns:
            List of DenseSearchResult sorted in descending score order (1-based ranks).
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query string cannot be empty or blank.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got: {top_k}")

        if len(self.document_ids) == 0:
            return []

        if self.encoder is None:
            model_id = self.manifest.get("model_id", DEFAULT_MODEL_ID)
            model_rev = self.manifest.get("model_revision", DEFAULT_MODEL_REVISION)
            self.encoder = ContrieverEncoder(
                model_id=model_id,
                model_revision=model_rev,
                device="cpu",
            )

        query_vec = self.encoder.encode_query(query.strip())
        if not isinstance(query_vec, np.ndarray):
            query_vec = np.asarray(query_vec, dtype=np.float32)

        if query_vec.ndim != 1 or query_vec.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"Query embedding dimension mismatch: expected "
                f"({self.embeddings.shape[1]},), got {query_vec.shape}."
            )
        if not np.isfinite(query_vec).all():
            raise ValueError("Query embedding contains non-finite values.")

        # Exact dot-product similarity
        scores = np.dot(self.embeddings, query_vec)

        # Deterministic descending score ordering
        k = min(top_k, len(self.document_ids))
        # Use argsort on negated scores with tie-breaking
        ranked_indices = np.argsort(-scores, kind="mergesort")[:k]

        results: list[DenseSearchResult] = []
        for rank_idx, doc_idx in enumerate(ranked_indices, start=1):
            doc_id = self.document_ids[doc_idx]
            score_val = float(scores[doc_idx])
            doc_record = self.corpus_documents[doc_id]
            title = str(doc_record.get("title", "")) if doc_record.get("title") else ""
            text = str(doc_record.get("text", ""))
            metadata = (
                doc_record.get("metadata", {})
                if isinstance(doc_record.get("metadata"), dict)
                else {}
            )

            results.append(
                DenseSearchResult(
                    document_id=doc_id,
                    title=title,
                    text=text,
                    score=score_val,
                    rank=rank_idx,
                    metadata=metadata,
                )
            )

        return results
