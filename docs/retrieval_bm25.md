# BM25 Sparse Lexical Retrieval

## 1. What BM25 Does

BM25 (Best Matching 25) is a sparse lexical retrieval algorithm derived from the probabilistic relevance framework. It ranks document passages based on the term frequencies of query words appearing in each document, balanced by inverse document frequency (penalizing ubiquitous words) and document length normalization (penalizing unnaturally long texts).

## 2. Why Nool Uses BM25 as Its First Retrieval Method

1. **Precision on Exact Terms**: Enterprise queries often involve specific acronyms, proper names, product IDs, and policy titles where exact token matching is critical.
2. **Computational Efficiency**: BM25 requires no GPU hardware or heavy neural networks. It builds rapidly on standard CPU architecture and evaluates queries in milliseconds.
3. **Lexical Baseline**: BM25 establishes the foundation for Nool's retrieval pipeline, which will subsequently be combined with dense embeddings (Contriever) into a hybrid retriever.

## 3. How Title and Text Are Indexed

Each passage in the prepared HotpotQA corpus contains an `_id`, a `title`, `text`, and optional `metadata`. During index construction:
- Searchable text is assembled by combining the passage title and body: `f"{title} {text}".strip()`.
- If a title is empty or absent, the text alone is indexed.
- The original discrete fields (`_id`, `title`, `text`, `metadata`) are preserved verbatim in an aligned document store (`documents.jsonl`) saved alongside the index.

## 4. Retrieval Settings and Tokenization

Nool uses the `bm25s` library (`version 0.3.16`) with explicit lower-level configuration:

| Setting | Value | Description |
|---|---|---|
| Method | `lucene` | BM25 variant scoring algorithm matching Lucene's implementation |
| $k_1$ | `1.5` | Controls term frequency saturation |
| $b$ | `0.75` | Controls document length normalization penalty |
| Tokenization | English stopwords | Standard English stopwords are removed during tokenization |
| Stemming | Disabled (`None`) | Preserves exact token surfaces without Porter/Snowball stemming |
| Indexed Fields | `title`, `text` | Document attributes compiled into searchable content |

The identical tokenization rules are applied to both indexed documents and search queries.

## 5. How the Index Is Saved and Loaded

The index is persisted in a local directory (`indexes/hotpotqa_sample_bm25/`):

- **BM25S Index Arrays**: `data.csc.index.npy`, `indices.csc.index.npy`, `indptr.csc.index.npy`, `vocab.index.json`, `params.index.json`.
- **Aligned Documents (`documents.jsonl`)**: JSONL storage of all passages in the exact positional sequence corresponding to index document IDs.
- **Index Manifest (`index_manifest.json`)**: Metadata recording schema version, library version, parameter values, document count, source corpus checksum, and creation timestamp.

Loading the index (`BM25Retriever.load(index_dir)`) restores the model, documents, and manifest into memory without requiring access to the original source corpus or archive.

## 6. How to Build and Search from the Command Line

### Building the Index

Build the index over the prepared HotpotQA sample:

```bash
python scripts/build_bm25_index.py
```

Optional arguments:
- `--corpus <path>`: Override path to source corpus JSONL.
- `--index-dir <path>`: Override destination index directory.
- `--overwrite`: Allow overwriting an existing index directory.

### Searching the Index

Search the persisted index with a query string:

```bash
python scripts/search_bm25.py --query "What company created the House of Cosbys?"
```

Retrieve a specific number of top results (e.g., top 5):

```bash
python scripts/search_bm25.py --query "Where was the World of Color show created?" --top-k 5
```

## 7. Structure of a Search Result

Each result is returned as a typed `BM25SearchResult` record:

```python
@dataclass(frozen=True)
class BM25SearchResult:
    document_id: str  # Original corpus document identifier (_id)
    title: str  # Document title
    text: str  # Passage body text
    score: float  # Non-negative BM25 relevance score
    rank: int  # 1-based rank in descending score order
    metadata: dict[str, Any]  # Preserved source metadata (e.g. URLs)
```

Queries that yield no matching tokens or only non-positive scores safely return an empty result list `[]`.

## 8. Dataset Scope and Limitations

> [!NOTE]
> This BM25 index is built over the reproducible 50,000-passage HotpotQA sample generated in Task 2 (`data/processed/hotpotqa_sample/corpus.jsonl`), not the full 5.2-million-passage BEIR corpus.
>
> Evaluation metrics and accuracy comparisons across models will be measured in subsequent benchmarking tasks.
