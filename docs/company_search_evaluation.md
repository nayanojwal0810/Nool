# Nool — Enterprise Knowledge Assistant: Company Search Evaluation

## What was checked

We searched the 45 fictional Lindenmere Services documents using 280 current passages. The question set has 150 cases with known source documents: 45 direct, 40 reworded, 40 needing two documents, and 25 near misses. The added questions cover exceptions, boundary conditions, and work that crosses teams. A case passes when every expected source document appears among the first five distinct documents. We also checked seven employee questions aimed at restricted documents.

These questions were written for this collection. The results show how this demo search behaves; they do not predict performance on an unrelated company collection or measure answer quality.

## Results

| Search method | Source cases passed | Rate |
|---|---:|---:|
| BM25 | 133 / 150 | 88.7% |
| Contriever | 141 / 150 | 94.0% |
| Hybrid | 142 / 150 | 94.7% |

Hybrid passed 44 of 45 direct cases, 39 of 40 reworded cases, 34 of 40 two-document cases, and all 25 near-miss cases. Its missed case IDs were `Q037`, `Q061`, `Q070`, `Q077`, `Q080`, `Q138`, `Q140`, and `Q141`. None of the seven restricted target documents appeared in the employee's top-five results. Four permission changes removed access to their target documents, and four document updates blocked the outdated index as expected.

The search returns one best passage per document so repeated passages from one source do not fill the result list. Every returned passage is checked against the user's current demo role and the document's current version and hash. A content update blocks use of the old index until it is rebuilt.

## Limits

The update and permission-change cases are safety checks, not part of the 150-case source retrieval rate. The five unsupported questions are covered in the separate [answer evaluation](company_answer_evaluation.md). These search checks do not show that the chosen passage contains every detail needed for an answer. The demo user selector is not production authentication.

Run `scripts/evaluate_company_search.py` after installing the company index to reproduce the source retrieval check. The script writes detailed counts and failed case IDs to `outputs/company_retrieval_evaluation.json`, which is excluded from Git.
