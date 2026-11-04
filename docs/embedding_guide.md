# Contriever embedding guide

Contriever represents each passage as a list of 768 numbers. Nool generated these passage vectors for all 50,000 sampled HotpotQA passages in Google Colab, then verified the downloaded files against the local corpus. Query vectors are generated when searching.

## Prepare the artifact

The [Colab notebook](../notebooks/contriever_embeddings_colab.ipynb) uses the original `facebook/contriever` model at revision `2bd46a25019aeea091fd42d1f0fd4801675cf699`. It joins each passage title and text, applies masked mean pooling, and saves `float32` vectors. Search uses dot product without vector normalization.

Upload `data/processed/hotpotqa_sample/corpus.jsonl` to `MyDrive/Nool/input/` and select a Colab GPU runtime. The notebook checks the corpus checksum before work begins. It saves 20 shards of 2,500 passages to Drive so an interrupted run can resume, then packages the vectors, document IDs, and manifest into a ZIP.

Extract `passage_embeddings.npy`, `document_ids.json`, and `index_manifest.json` into `indexes/contriever_hotpotqa_sample/`. The row order of the vectors must match the document ID order. Run the local verifier before searching:

```powershell
.\.venv\Scripts\python scripts/verify_contriever_artifact.py
```

The current local artifact passed verification for 50,000 passages, 768 dimensions, `float32`, and the recorded corpus and artifact checksums. The files are excluded from Git because they are large.

Install the dense dependencies before running a search:

```powershell
.\.venv\Scripts\python -m pip install -e ".[dense]"
.\.venv\Scripts\python scripts/search_contriever.py --query "Where was Marie Curie born?"
```

The [Contriever model](https://huggingface.co/facebook/contriever) is published under CC BY-NC 4.0. Its terms should be reviewed before commercial use.
