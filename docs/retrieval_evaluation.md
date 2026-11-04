# Nool — Retrieval Benchmark Evaluation (BM25 vs Contriever)

## 1. Why We Need a Benchmark

In enterprise question-answering systems, retrieval is the critical foundation. If the retrieval step fails to surface the relevant passages from documentation, even the best language models will either hallucinate or state that the answer is unknown.

Nool implements two distinct retrieval paradigms:
1. **BM25**: A sparse lexical retriever based on term frequency, inverse document frequency (IDF), and document-length normalization.
2. **Contriever**: A dense semantic retriever using pre-trained neural transformer embeddings trained via contrastive learning.

To make informed engineering decisions—and ultimately combine both in hybrid retrieval—we need an objective, reproducible benchmark comparing both systems on the exact same corpus, queries, and ground-truth relevance judgments.

---

## 2. What the Retrievers Do

### BM25 (Sparse Lexical Retrieval)
- **Mechanism**: Analyzes exact token matches between user queries and corpus documents.
- **Strengths**: Excels at finding specific keywords, proper nouns, unique names, acronyms, product identifiers, and technical terminology.
- **Limitations**: Suffers from the "vocabulary mismatch" problem—if the query uses different words from the document (e.g., "automobile" vs. "car"), BM25 cannot recognize their semantic similarity.

### Contriever (Dense Semantic Retrieval)
- **Mechanism**: Encodes documents and queries into 768-dimensional continuous vector embeddings. Retrieval is performed using exact dot-product similarity between the query embedding and precomputed passage vectors.
- **Strengths**: Captures semantic intent, synonyms, paraphrases, and context even when exact keywords do not match.
- **Limitations**: Computationally heavier at query time (requires neural forward pass) and can occasionally retrieve semantically similar passages that miss critical exact technical identifiers.

---

## 3. Benchmark Dataset Scope & Limitations

The evaluation uses the standardized, reproducible 50,000-passage sample generated from the HotpotQA (BEIR format) dataset:

- **Source Corpus**: `data/processed/hotpotqa_sample/corpus.jsonl`
  - Passages: Exactly 50,000 unique document IDs.
  - Checksum: `c3970d91ebfd36d794066b42bd7acb85a744aabba2c97dd9a9bfde52d3dcbfbf`
- **Queries**: `data/processed/hotpotqa_sample/queries.jsonl`
- **Relevance Judgments**:
  - Test Split: `data/processed/hotpotqa_sample/qrels/test.tsv` (500 unique test queries, 1,000 qrel rows).
  - Development Split: `data/processed/hotpotqa_sample/qrels/dev.tsv` (500 unique dev queries, 1,000 qrel rows).
- **Default Split**: `test` only. Development queries are never evaluated or mixed into test evaluations.

> [!IMPORTANT]
> **Comparison with Full-Corpus Benchmarks**:
> The 50,000-passage benchmark sample is smaller than the original 5,233,329-passage HotpotQA BEIR corpus. Because the background distractor pool is smaller, absolute recall and precision scores will naturally be higher than official full-corpus leaderboard scores.
>
> These numbers must **not** be compared directly to official 5.2M-passage HotpotQA benchmarks. The purpose of this benchmark is internal, comparative, and reproducible evaluation between Nool's retrieval methods under controlled local conditions.

---

## 4. Evaluation Metrics Explained

All metrics are computed at rank cutoffs $k \in \{1, 5, 10\}$ up to the standard evaluation cutoff of 10.

### Recall@k
- **Definition**: The proportion of all relevant documents for a query that appear in the top $k$ retrieved results.
- **Formula**:
  $$\text{Recall@}k = \frac{|\text{retrieved}_{1..k} \cap \text{relevant}|}{|\text{relevant}|}$$
- **Interpretation**: Higher Recall@10 indicates that fewer relevant evidence passages are completely missed.

### MRR@10 (Mean Reciprocal Rank)
- **Definition**: The reciprocal of the rank at which the first relevant document is found within the top 10 positions.
- **Formula**:
  $$\text{RR@}10 = \begin{cases} \frac{1}{\text{rank}_{\text{first}}} & \text{if } 1 \le \text{rank}_{\text{first}} \le 10 \\ 0 & \text{otherwise} \end{cases}$$
- **Interpretation**: Measures how quickly a user or language model encounters the first useful piece of evidence.

### nDCG@10 (Normalized Discounted Cumulative Gain)
- **Definition**: Measures ranking quality with graded relevance and logarithmic rank discount.
- **Formulas**:
  $$\text{Gain}_i = 2^{\text{relevance}_i} - 1$$
  $$\text{Discount}_i = \log_2(i + 1)$$
  $$\text{DCG@}10 = \sum_{i=1}^{10} \frac{\text{Gain}_i}{\text{Discount}_i}$$
  $$\text{nDCG@}10 = \frac{\text{DCG@}10}{\text{IDCG@}10}$$
- **Interpretation**: Rewards ranking highly relevant documents at higher positions rather than near the bottom of the list.

### AP@10 & MAP@10 (Mean Average Precision)
- **Definition**: Evaluates precision at every rank where a relevant document is retrieved, normalized by $\min(|\text{relevant}|, 10)$.
- **Formula**:
  $$\text{AP@}10 = \frac{1}{\min(|\text{relevant}|, 10)} \sum_{i=1}^{10} \text{Precision@}i \cdot \mathbb{I}(\text{doc}_i \text{ is relevant})$$
  $$\text{MAP@}10 = \frac{1}{|Q|} \sum_{q \in Q} \text{AP@}10(q)$$
- **Interpretation**: Combines both precision and recall across multiple relevant documents per query.

### Macro-Aggregation
All benchmark numbers report the **macro-average** across all 500 test queries. Queries where a retriever returns zero relevant documents receive 0.0 and are never omitted from the average.

---

## 5. Latency Measurement

Search latency is measured per query using `time.perf_counter()`:
- **BM25 Latency**: Measures plaintext tokenization, index lookup, and score ranking.
- **Contriever Latency**: Measures query tokenization, transformer forward pass on device, mean pooling, and exact matrix-vector dot product scoring.
- **Warm-Up**: Both systems execute one warm-up query prior to the evaluation loop. This warm-up pass is excluded from latency and metric statistics.
- **Reported Statistics**: Mean, median, and 95th percentile (P95) latency in milliseconds.

> [!NOTE]
> Latency depends on host hardware (CPU model, memory speed, OS). It provides an internal relative comparison between sparse and dense retrieval on the host machine.

---

## 6. How to Run the Benchmark

From the project root:

```powershell
.\.venv\Scripts\python scripts/evaluate_retrieval.py
```

### Command Line Options

| Option | Default | Description |
|---|---|---|
| `--split` | `test` | Split to evaluate: `test` or `dev` |
| `--top-k` | `10` | Rank cutoff (must be $\ge 10$) |
| `--output-dir` | `outputs/retrieval_benchmark/<split>` | Output destination directory |
| `--corpus` | `configs/app.yaml` | Path to benchmark corpus JSONL |
| `--queries` | `data/processed/hotpotqa_sample/queries.jsonl` | Path to queries JSONL |
| `--qrels` | `data/processed/hotpotqa_sample/qrels/<split>.tsv` | Path to qrels TSV |
| `--bm25-index` | `configs/app.yaml` | Path to BM25 index directory |
| `--contriever-artifact` | `configs/app.yaml` | Path to Contriever artifact directory |
| `--device` | `cpu` | Inference device (`cpu`, `cuda`, `auto`) |

---

## 7. Inspectable Output Files

All evaluation artifacts are saved in `outputs/retrieval_benchmark/test/`:

1. **`summary.json`**:
   Comprehensive JSON summary containing:
   - Evaluated split and query counts.
   - Verified corpus SHA-256 and document counts.
   - BM25 and Contriever configurations.
   - Aggregate retrieval metrics and latency percentiles.
   - Timestamp and Git commit SHA.

2. **`comparison.csv`**:
   Clean tabular comparison between BM25 and Contriever across all standard metrics:
   ```csv
   Metric,BM25,Contriever
   Recall@1,...
   Recall@5,...
   Recall@10,...
   MRR@10,...
   nDCG@10,...
   MAP@10,...
   Mean latency (ms),...
   Median latency (ms),...
   P95 latency (ms),...
   ```

3. **`per_query_results.csv`**:
   Detailed breakdown with one row per query:
   - Query ID and text.
   - Expected relevant document IDs.
   - BM25 and Contriever retrieved document IDs and ranks.
   - Per-query Recall@1, 5, 10, MRR@10, nDCG@10, and AP@10.
   - Per-query search latency.

This per-query data is useful for error analysis—identifying queries where lexical retrieval succeeds but dense retrieval fails, and vice versa.
