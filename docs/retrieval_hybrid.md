# Hybrid retrieval evaluation

Nool's main retriever combines BM25 keyword search and Contriever semantic search with reciprocal rank fusion (RRF). Each retriever returns its top 10 passages. A passage's fusion score is the sum of `1 / (20 + rank)` from the lists where it appears. Shared passages are counted once in the final ranking. The final top 10 are passed to the next stage of the system.

The rank constant 20 was selected from 20, 60, and 100 using the 500 development queries. Selection prioritized Recall@10, then nDCG@10. The setting was fixed before the 500 test queries were evaluated. The test comparison used the saved BM25 and Contriever rankings produced on the same 50,000-passage corpus; it did not regenerate embeddings or make another sample.

| Test metric | BM25 | Contriever | Hybrid |
|---|---:|---:|---:|
| Recall@1 | 0.462 | 0.461 | **0.480** |
| Recall@5 | 0.815 | 0.779 | **0.851** |
| Recall@10 | 0.859 | 0.825 | **0.882** |
| MRR@10 | 0.9447 | 0.9433 | **0.9742** |
| nDCG@10 | 0.8359 | 0.8107 | **0.8729** |
| MAP@10 | 0.7730 | 0.7417 | **0.8214** |

The hybrid ranking improves all listed test metrics against both individual retrievers on this fixed sample. Its search requires both retrievers. The prior benchmark measured average BM25 search at 3.55 ms and Contriever at 177.33 ms on the recorded CPU; combining saved rankings adds little computation, but this evaluation did not measure an end-to-end hybrid request. We will measure that when the application is integrated.

At the query level, hybrid Recall@10 was higher than BM25 on 32 questions, lower on 10, and equal on 458. Against Contriever it was higher on 61, lower on 4, and equal on 435. This is useful evidence of complementary results, while also showing that fusion can hurt some individual questions.

These numbers describe retrieval on a sampled HotpotQA corpus. They do not evaluate enterprise permissions, evidence reuse, or answer correctness, and they are not comparable to full-corpus BEIR leaderboard results. The sample retains the judged positive passages but may contain unjudged passages that are also useful.

To reproduce from a fresh checkout with the verified local corpus and indexes in place, run `scripts/evaluate_retrieval.py` for both `--split dev` and `--split test` in the project virtual environment. Then run `scripts/evaluate_hybrid.py --split dev` followed by `--split test`. Both hybrid commands validate the fixed corpus and qrels before evaluating. The generated selection, summaries, and per-query rankings are saved under `outputs/hybrid_benchmark/` and are excluded from Git.
