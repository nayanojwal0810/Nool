# Lindenmere Services document collection

Lindenmere Services is a fictional company used to exercise Nool's document and access workflows. The collection contains 45 current documents: 39 Markdown files, three text files, and three text-based PDFs. They cover people policies, IT, finance, onboarding, operations, and restricted administrator procedures. Names and rules are fictional; the documents are not legal or employment advice.

`manifest.yaml` lists each document's stable ID, title, source file, owner, version, effective date, and allowed role. The source files use the same identifying details. Employees may access 38 documents. Four are for HR administrators and three are for IT administrators. The three known demo users are `employee_demo`, `hr_demo`, and `it_admin_demo`.

`revisions/` contains four later policy versions for update demonstrations. `question_cases.jsonl` contains 170 cases with an expected answer or abstention, source document IDs, demo user, and scenario. Of these, 150 test source finding: 45 direct, 40 reworded, 40 needing two sources, and 25 near misses. The remaining cases cover restricted access, document updates, permission changes, and unsupported questions. The cases define a controlled evaluation set; they are not evidence that the application has passed those scenarios yet.

Run `scripts/ingest_demo_documents.py` from the project root to validate and load the current collection into a local SQLite database. The database is excluded from Git. Ingestion keeps previous passages when a source changes and exposes only the current active version through `current_chunks`. The company search and answer paths check permissions before using passages. The website is still in development.

Use the [company search Colab notebook](../../notebooks/company_search_colab.ipynb) to build the BM25 and Contriever indexes with a GPU, check source retrieval, and save the indexes as a ZIP. Run `scripts/search_company.py "How many annual leave days do I get?" --user employee_demo` after extracting the ZIP into `indexes/`. The search path uses the current role record, rejects outdated indexes after a document update, and returns no restricted passages. Rebuild the indexes after a content update. A role change takes effect without rebuilding. The demo user names are test identities, not authentication for a public website.

## Apply document changes

Edit a source file or its allowed roles in `manifest.yaml`, then run `scripts/sync_demo_documents.py`. The command updates the local SQLite record, archives documents removed from the manifest, and reports whether the search index needs rebuilding. Role-only changes take effect immediately. A changed, archived, restored, or renamed document blocks the old index until a new one is built.

For a rebuild, upload `data/processed/demo_documents.sqlite` to `MyDrive/Nool/input/company_documents.sqlite`. In the company search notebook, set `USE_UPDATED_DATABASE = True` and run its cells on a Colab GPU. Download the new ZIP and extract it into `indexes/`. Keep the local SQLite database; its versions must match the new index. The original 150-question retrieval score describes the original documents and does not apply after their content changes.
