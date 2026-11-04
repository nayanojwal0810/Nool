# Nool

Nool is an enterprise knowledge assistant being built to answer questions from company documents and show its sources. Its main feature is safe evidence reuse: before using a passage found for an earlier question, Nool checks that the source is still current and the user can still access it.

## How it works

Nool's main search combines BM25, which matches words, and Contriever, which matches meaning. The planned application will check saved evidence first, search documents when needed, validate selected passages, and use them to write a cited answer. If the documents do not support an answer, it will say so.

The application is planned in Python with a Streamlit website, SQLite for document and evidence records, and Qwen through Ollama for answers.

Search is working now. Document management, access checks, evidence memory, answer generation, and the Streamlit website are still in development. The website is intended for online demonstration with fictional company documents.

## Retrieval results

We tested all three search methods on the same 50,000 HotpotQA passages and 500 test questions.

| Method | Recall@10 | nDCG@10 |
|---|---:|---:|
| BM25 | 0.859 | 0.836 |
| Contriever | 0.825 | 0.811 |
| Hybrid | 0.882 | 0.873 |

These results measure search on a fixed sample. They do not measure answer quality or document security.

## Run the current code

Use Python 3.11. Install the project and run its tests:

```powershell
git clone https://github.com/nayanojwal0810/Nool.git
cd Nool
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest
```

Contriever search also needs the dense dependencies and a local copy of the passage embeddings. The [embedding guide](docs/embedding_guide.md) explains how to prepare them. Large data files, model weights, and indexes are excluded from Git.

## Project guides

- [System overview](docs/system_overview.md): current code and intended application.
- [Dataset](docs/dataset.md): source data, fixed sample, and integrity checks.
- [Retrieval evaluation](docs/retrieval_evaluation.md): methods, results, and limits.
- [Embedding guide](docs/embedding_guide.md): Colab generation and local verification.
