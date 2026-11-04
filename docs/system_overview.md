# System overview

Nool is being built as a website where employees ask questions about company documents and see the sources behind each answer. It may reuse a passage found for an earlier question, but only after checking that the source is still valid for the current user.

## Current system

The working code prepares a fixed HotpotQA sample and searches it with BM25, Contriever, or their combined ranking. BM25 finds matching words. Contriever turns text into vectors and finds passages with similar meaning. Reciprocal rank fusion combines their result positions into Nool's main search ranking. The retrieval benchmark measures how often each method finds known relevant passages.

HotpotQA is used for search evaluation. It does not represent company documents or test access rules. Nool does not yet have a working question-to-answer website.

## Intended application

The website will use fictional company documents and a small set of demo users. A question will follow this path:

1. Resolve the user's current access rights.
2. Look for relevant passages in evidence memory.
3. Check each saved passage against the document's current status, version, content, and permissions.
4. If valid evidence is insufficient, run fresh hybrid search and check those results too.
5. Send only approved passages to the answer model and return an answer with source citations. If the passages do not support an answer, say so.
6. Save useful evidence for later questions.

Document updates and permission changes will take effect before evidence is reused. The demo user selector will simulate identities; it will not serve as production authentication. The website's hosting and model runtime will be tested when the application is built.

## Code layout

`src/nool/data` prepares the public search sample. `src/nool/retrieval` contains BM25, Contriever, and hybrid search. `src/nool/evaluation` checks data and calculates search metrics. `configs/app.yaml` stores current settings. The `evidence`, `security`, `generation`, and `app` packages are reserved for the intended application.
