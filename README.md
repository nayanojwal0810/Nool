# Nool — Enterprise Knowledge Assistant

Employees often search several policies and guides to answer a simple workplace question. Nool is being built to find the right passage, answer from it, and show its source. When someone asks a similar question later, Nool can reuse the passage after checking that the document is still current and the employee still has access.

## How it works

Nool's main search combines BM25, which matches words, and Contriever, which matches meaning. The planned application will check saved evidence first. It will search documents when needed, check the selected passages against current document and access records, and write an answer with citations. It will say when the documents do not support an answer.

The application will use Python, a Streamlit website, SQLite for document and evidence records, and Qwen through Ollama for answers.

Search is working now. Document management, access checks, evidence memory, answer generation, and the website are still in development. The online demonstration will use realistic fictional company documents.

## Retrieval results

We tested all three search methods on the same 50,000 HotpotQA passages and 500 test questions. Recall@10 measures how many known relevant passages appear in the first 10 results. nDCG@10 also accounts for their order.

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
