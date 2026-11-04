# Nool

An enterprise knowledge assistant that answers employee questions from company documents and safely reuses verified evidence.

## Overview

Nool is a local-first enterprise knowledge assistant designed to help employees find accurate answers from internal documentation, including HR policies, IT support guides, and operational procedures. The system focuses on reliable evidence retrieval, access control verification, document freshness checking, and grounded answer generation with clear citations.

## Problem Being Addressed

Internal company knowledge is frequently fragmented across separate policies, handbooks, and intranet portals. Employees spend significant time locating relevant documentation, and existing automated assistants often generate ungrounded answers, ignore permission boundaries, or rely on outdated revisions of company documents. Nool addresses these challenges by maintaining an auditable evidence pipeline that validates document permissions, freshness, and relevance before generating responses.

## Intended Answer Flow

1. **Query Processing**: An employee submits a question through the user interface.
2. **Hybrid Retrieval**: The retrieval engine searches indexed company documents using sparse lexical matching and dense semantic embeddings.
3. **Evidence Validation**: Candidate passages are verified against access permissions, document currency (freshness), and relevance.
4. **Evidence Memory**: Reusable evidence items are stored in an indexed evidence memory for efficient matching on related queries.
5. **Answer Generation**: A local language model generates a concise response grounded strictly in the validated evidence passages, citing all relevant sources.

## Main Planned Capabilities

- **Hybrid Retrieval (Implemented)**: BM25 and Contriever rankings are combined with reciprocal rank fusion on the fixed HotpotQA sample.
- **Evidence Memory (Planned)**: Fast vector search indexing (HNSW) and SQLite storage for persistent, reusable evidence records.
- **Access and Freshness Validation (Planned)**: Policy-based filtering to ensure users only see information permitted by their role and drawn from current document versions.
- **Local Generation (Planned)**: Grounded response generation using a local Qwen language model without sending enterprise data to external APIs.
- **Interactive Interface (Planned)**: Streamlit web application providing question-answering, citation inspection, and administrative overview.
- **Evaluation Suite (Planned)**: Automated benchmarks measuring retrieval precision, recall, and answer faithfulness.

## Technology Stack

- **Language**: Python 3.11
- **Configuration & Data Handling**: PyYAML, JSONL
- **Search & Retrieval**: BM25, Contriever, reciprocal rank fusion
- **Vector & Evidence Storage (Planned)**: HNSW, SQLite
- **Model Execution (Planned)**: Local Qwen model
- **User Interface (Planned)**: Streamlit
- **Testing & Quality**: pytest, pytest-cov, Ruff

## Project Structure

```text
Nool/
├── .github/
│   └── workflows/
│       └── quality.yml  # Quality and CI workflow
├── configs/
│   └── app.yaml         # Application configuration
├── data/
│   ├── raw/             # Unprocessed source data and archives
│   ├── interim/         # Partially processed intermediate data
│   ├── processed/       # Final tokenized and indexed datasets
│   └── manifests/       # Dataset provenance and checksum records
├── docs/
│   └── system_overview.md # Architectural documentation
├── notebooks/           # Exploratory analysis notebooks
├── scripts/             # Operational and data processing scripts
├── src/
│   └── nool/            # Core package source code
│       ├── config/      # Configuration loading and path resolution
│       ├── data/        # Ingestion and preprocessing pipelines
│       ├── retrieval/   # Lexical and dense retrieval components
│       ├── evidence/    # Evidence memory and validation
│       ├── security/    # Access control and permission filtering
│       ├── generation/  # Answer synthesis and prompt templates
│       ├── evaluation/  # Retrieval and generation metrics
│       └── app/         # Web interface implementation
└── tests/               # Unit, integration, and fixture tests
    ├── unit/            # Unit tests
    ├── integration/     # Integration tests
    └── fixtures/        # Test fixtures and mock data
```

## Setup

### Prerequisites

- Python 3.11
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nayanojwal0810/Nool.git
   cd Nool
   ```

2. Create and activate a Python 3.11 virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. Install the package in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Run tests and lint checks:
   ```bash
   ruff check .
   ruff format --check .
   pytest
   ```

## Current Implementation Status

The retrieval foundation is implemented:
- Repository layout and directory structure configured.
- Packaging metadata (`pyproject.toml`) and Python 3.11 environment defined.
- Application configuration (`configs/app.yaml`) and configuration loader created.
- Quality control toolchain (Ruff, pytest) and CI workflow established.
- Initial unit test suite passing.
- Raw dataset inspection and manifest generation completed.
- BM25, Contriever, and hybrid retrieval run on the same 50,000-passage sample. The test results and limitations are in the [hybrid evaluation](docs/retrieval_hybrid.md).
- Enterprise document ingestion, evidence memory, permission and freshness validation, model generation, and the Streamlit website remain to be implemented.
