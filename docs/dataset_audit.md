# HotpotQA Dataset Audit & Evaluation Sample

## 1. Dataset Purpose

HotpotQA (in BEIR format) provides a benchmark for multi-hop retrieval and question-answering evaluation. In Nool, this dataset serves as the benchmark ground truth for evaluating retrieval quality (lexical BM25, dense Contriever, and hybrid retrieval) under controlled evidence memory conditions.

## 2. Dataset Structure

The raw benchmark is distributed as a single ZIP archive (`data/raw/hotpotqa.zip`) containing:

- `hotpotqa/corpus.jsonl`: Document passages formatted as JSON lines (`_id`, `title`, `text`, optional `metadata`).
- `hotpotqa/queries.jsonl`: Question records formatted as JSON lines (`_id`, `text`, optional `metadata`).
- `hotpotqa/qrels/dev.tsv`: Tab-separated relevance judgments for development queries (`query-id`, `corpus-id`, `score`).
- `hotpotqa/qrels/test.tsv`: Tab-separated relevance judgments for test queries (`query-id`, `corpus-id`, `score`).
- `hotpotqa/qrels/train.tsv`: Tab-separated relevance judgments for training queries.

## 3. How to Inspect the Archive

Run the factual audit tool from the project root:

```bash
python scripts/audit_hotpotqa.py
```

Optional parameters:
- `--archive <path>`: Path to target zip file (defaults to `data/raw/hotpotqa.zip`).
- `--manifest <path>`: Manifest destination (defaults to `data/manifests/hotpotqa.yaml`).

The command inspects the archive structure, validates integrity, counts records across all members via incremental streaming, verifies ID and qrel consistency, and writes a verified manifest without extracting the 2.1 GB corpus to disk.

## 4. Actual Audit Results

Audit performed on `data/raw/hotpotqa.zip`:

| Metric | Inspected Value |
|---|---|
| Archive File Size | 654,025,350 bytes (~623.73 MB) |
| SHA-256 Checksum | `c62459dbc91d31329584507b1fa564b3b1abcee58800967113054ee0f5a15c62` |
| ZIP Integrity | Verified valid (zero archive corruption) |
| Total Corpus Passages | 5,233,329 |
| Total Queries | 97,852 |
| Dev Qrels | 10,894 rows (5,447 unique queries) |
| Test Qrels | 14,810 rows (7,405 unique queries) |
| Train Qrels | 170,000 rows (85,000 unique queries) |
| Duplicate Query IDs | 0 |
| Missing Query References | 0 |
| Missing Corpus References | 0 |
| Malformed Records | 0 |

## 5. How to Generate the 50,000-Passage Sample

Execute the reproducible sampling script:

```bash
python scripts/prepare_hotpotqa_sample.py
```

Configurable arguments:
- `--archive`: Path to raw ZIP archive (default: `data/raw/hotpotqa.zip`).
- `--output-dir`: Output destination (default: `data/processed/hotpotqa_sample`).
- `--corpus-sample-size`: Sampled passage count (default: `50000`).
- `--dev-queries`: Sampled development queries (default: `500`).
- `--test-queries`: Sampled test queries (default: `500`).
- `--seed`: Random seed (default: `42`).
- `--manifest`: Manifest destination (default: `data/manifests/hotpotqa_sample.yaml`).

### Output Structure

The generated dataset is written to `data/processed/hotpotqa_sample/`:

```text
data/processed/hotpotqa_sample/
├── corpus.jsonl       # Exactly 50,000 passages (positive + background)
├── queries.jsonl      # Exactly 1,000 queries (500 dev + 500 test)
└── qrels/
    ├── dev.tsv        # All relevance judgments for the 500 dev queries
    └── test.tsv       # All relevance judgments for the 500 test queries
```

### Actual Sample Statistics (Default Settings: Seed 42)

- Development Queries: 500 (1,000 qrels)
- Test Queries: 500 (1,000 qrels)
- Total Selected Queries: 1,000 (disjoint dev and test sets)
- Positive Relevant Passages: 1,981 (all preserved)
- Background / Negative Passages: 48,019 (streaming reservoir sampled)
- Total Sample Corpus Passages: 50,000

## 6. Sampling Limitations

> [!IMPORTANT]
> The 50,000-passage corpus is a sampled evaluation corpus, not the official full-corpus BEIR evaluation.
>
> 1. **Candidate Pool Size**: The full HotpotQA BEIR benchmark contains 5,233,329 passages. Retrieval metrics computed on this 50,000-passage sample will yield higher absolute recall numbers than a full-corpus benchmark because the distractor pool is smaller.
> 2. **Evaluation Purpose**: This sample is designed for rapid, reproducible local experiments, CI testing, and early retrieval parameter tuning.
> 3. **Positive Completeness**: All ground-truth positive passages for the 1,000 selected queries are guaranteed present, ensuring fair comparative evaluation across retrieval models.
