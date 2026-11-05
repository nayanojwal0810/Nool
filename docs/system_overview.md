# Nool — Enterprise Knowledge Assistant: System Overview

Company policies change, and employees have different access rights. A saved answer or passage can become wrong or restricted after an update. Nool is being built as a website where employees ask questions about company documents and see the sources behind each answer. It may reuse a passage found for an earlier question only after checking the current document and the current user's access.

## Current system

The working code prepares a fixed HotpotQA sample and searches it with BM25, Contriever, or their combined ranking. BM25 finds matching words. Contriever turns text into vectors and finds passages with similar meaning. Reciprocal rank fusion combines their result positions into Nool's main search ranking. The retrieval benchmark measures how often each method finds known relevant passages.

The [Lindenmere Services collection](../data/demo/README.md) adds 45 fictional company documents and 170 prepared question cases. Ingestion reads Markdown, plain text, and text-based PDFs. SQLite stores each document's current version, hash, status, allowed roles, and traceable passages. The company search path combines BM25 and Contriever passage rankings, checks the live role and document record, and blocks search if its index is outdated. The [company search evaluation](company_search_evaluation.md) reports results on the current collection. The [answer path](answer_guide.md) sends approved passages to a local model and checks its cited excerpts. Real model answers have not yet been measured.

HotpotQA is used for search evaluation. It does not represent company documents or test access rules. Nool does not yet have a working question-to-answer website.

## Intended application

The website will use the 45 fictional company documents and known demo users. A question will follow this path:

1. Resolve the user's current access rights.
2. Look for relevant passages in evidence memory.
3. Check each saved passage against the document's current status, version, content, and permissions.
4. If valid evidence is insufficient, run fresh hybrid search and check those results too.
5. Send only approved passages to the answer model and return an answer with source citations. If the passages do not support an answer, say so.
6. Save useful evidence for later questions.

The document store will keep a source ID, version, content hash, status, and allowed roles for each document. A passage will retain its source and location so a citation can point back to it. Document updates and permission changes will take effect before evidence is reused. The demo user selector will simulate identities; it will not serve as production authentication. The website's hosting and model runtime will be tested when the application is built.

The 170 prepared questions cover direct answers, reworded questions, missing answers, document updates, and access changes. Each case names the expected source and outcome. The 150 source-finding cases test retrieval now; other cases will test application behavior that HotpotQA cannot test once the request flow is built.

## Code layout

`src/nool/data` prepares the public search sample. `src/nool/retrieval` contains BM25, Contriever, and hybrid search. `src/nool/evaluation` checks data and calculates search metrics. `configs/app.yaml` stores current settings. The `evidence`, `security`, `generation`, and `app` packages are reserved for the intended application.
