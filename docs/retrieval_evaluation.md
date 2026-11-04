# Retrieval evaluation

Search must find useful passages before Nool can answer a question. We compared three methods on the same 50,000-passage HotpotQA sample and 500 test questions. BM25 matches words, Contriever matches meaning, and Nool's main hybrid search combines their ranked lists.

## Methods

BM25 uses `bm25s` with Lucene scoring, `k1=1.5`, `b=0.75`, English stopwords, and no stemming. It indexes each passage's title and text.

Contriever uses the pinned `facebook/contriever` model. The 50,000 passage vectors were generated in Colab and checked locally. Each query uses the same model, and the code ranks passages by dot product.

Hybrid search takes the top 10 results from each method and adds `1 / (20 + rank)` for each passage's positions. It removes duplicate passages and returns the 10 highest scores. The development questions were used to set this value; the test questions produced the results below.

## Test results

| Metric | BM25 | Contriever | Hybrid |
|---|---:|---:|---:|
| Recall@1 | 0.462 | 0.461 | 0.480 |
| Recall@5 | 0.815 | 0.779 | 0.851 |
| Recall@10 | 0.859 | 0.825 | 0.882 |
| MRR@10 | 0.9447 | 0.9433 | 0.9742 |
| nDCG@10 | 0.8359 | 0.8107 | 0.8729 |
| MAP@10 | 0.7730 | 0.7417 | 0.8214 |

Recall@10 is the share of known relevant passages found in the first 10 results. MRR rewards finding a relevant passage early. nDCG and MAP also measure ranking quality. Scores are averaged across all 500 test questions.

Hybrid Recall@10 improved over BM25 on 32 questions, fell on 10, and tied on 458. It improved over Contriever on 61, fell on 4, and tied on 435. The combined ranking helps overall, although it does not help every question.

The saved CPU benchmark measured mean search time of 3.55 ms for BM25 and 177.33 ms for Contriever. Hybrid search runs both. We have not yet measured the complete hybrid request time or an end-to-end answer time.

## Reproduce the comparison

Prepare the fixed sample and both local indexes first. Then run these commands from the project root using the project Python environment:

```powershell
.\.venv\Scripts\python scripts/evaluate_retrieval.py --split dev
.\.venv\Scripts\python scripts/evaluate_retrieval.py --split test
.\.venv\Scripts\python scripts/evaluate_hybrid.py --split dev
.\.venv\Scripts\python scripts/evaluate_hybrid.py --split test
```

The commands check corpus and document ID alignment. Detailed results are written under `outputs/`, which is excluded from Git. These scores describe retrieval on the fixed sample. They do not measure generated answers, evidence reuse, access control, or the full HotpotQA corpus.
