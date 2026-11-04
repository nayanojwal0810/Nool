# Contriever Dense Retrieval & Colab Embedding Workflow

## 1. Introduction: What Is Dense Retrieval?

In standard lexical search (such as BM25), documents are matched based on exact shared keywords. If a user asks *"Where was Marie Curie born?"*, a lexical retriever looks for documents that specifically contain the words *"Marie"*, *"Curie"*, and *"born"*.

**Dense retrieval** takes a semantic approach. It translates both documents and user questions into dense, fixed-dimensional numerical vectors called **embeddings**. Instead of matching exact strings, dense retrieval measures the geometric proximity between the question's vector and each document's vector in high-dimensional embedding space. This allows the system to discover conceptually relevant passages even when they use entirely different vocabulary, synonyms, or paraphrases.

---

## 2. What Is Contriever?

**Contriever** is an unsupervised dense information retrieval model developed by Meta AI Research ([Izacard et al., 2021](https://arxiv.org/abs/2112.09118)). It is based on a standard 12-layer BERT architecture (`bert-base-uncased`) with a hidden dimension of 768.

Contriever was trained without human annotations using **contrastive learning**: the model learned to map two augmented views of the same document close together in vector space while pushing representations of different documents apart. As a result, Contriever demonstrates strong zero-shot retrieval capabilities across diverse domains without requiring supervised fine-tuning.

### Model Selection & Verified Revision
- **Model ID**: `facebook/contriever`
- **Immutable Revision SHA**: `2bd46a25019aeea091fd42d1f0fd4801675cf699`
- **Dimension**: 768 float32 dimensions
- **Maximum Sequence Length**: 512 tokens
- **Excluded Variants**: `facebook/contriever-msmarco` and `facebook/mcontriever` are intentionally not used to ensure pure unsupervised zero-shot baseline performance.

---

## 3. Embedding Methodology

### A. Text Preparation
For every passage in the corpus, the text is formatted as:
```python
search_text = f"{title} {text}".strip()
```
- Both the document title and the passage body are incorporated.
- No artificial task instructions, prompts, or prefixes are added.
- Original casing and punctuation are preserved (the Hugging Face tokenizer handles subword tokenization).

### B. Attention-Mask-Aware Mean Pooling
Contriever produces token representations from its last hidden layer `(batch_size, sequence_length, 768)`. Because sentences have varying lengths and are padded with special padding tokens, simple averaging across all tokens would contaminate the sentence representation with padding vectors.

We use **attention-mask-aware mean pooling**:
1. Padding tokens are identified via the attention mask (`attention_mask == 0`).
2. Padding representations are zeroed out: `token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.0)`.
3. Valid token vectors are summed along the sequence axis.
4. The sum is divided by the count of non-padding tokens:
   $$\text{Embedding} = \frac{\sum_{i=1}^{T} (\text{Token}_i \cdot \text{Mask}_i)}{\max(\sum_{i=1}^{T} \text{Mask}_i, 10^{-9})}$$

### C. Normalization & Similarity
- **No L2 Normalization**: Raw Contriever embeddings are not normalized to unit length.
- **Dot-Product Similarity**: Passages are ranked using standard dot product:
  $$\text{Score}(q, d) = \mathbf{e}_q \cdot \mathbf{e}_d = \sum_{j=1}^{768} e_{q,j} \cdot e_{d,j}$$
- The same text preparation and pooling methodology is applied identically to passages and search queries.

---

## 4. Why Document IDs and Embeddings Must Remain Aligned

Dense retrieval stores embeddings as an anonymous numerical matrix of shape `(50000, 768)`. The matrix itself does not contain document IDs, titles, or text.

Row $i$ of `passage_embeddings.npy` corresponds strictly to document $i$ in `document_ids.json`. When dot-product scoring identifies that row index 42 has the highest score, Nool uses index 42 to look up the document ID in `document_ids.json`, which in turn retrieves the passage's title and text from `corpus.jsonl`. If rows are reordered or misaligned, the system will return incorrect passages.

---

## 5. Offline Passage Encoding vs. Online Query Encoding

- **Offline Passage Encoding (Colab GPU)**:
  Encoding 50,000 passages requires evaluating millions of transformer tokens. Running this on a consumer laptop CPU would take hours or days and consume significant battery and thermal headroom. We run this heavy batch job once in Google Colab using a free GPU.
- **Online Query Encoding (Local CPU)**:
  When an employee asks a single question, encoding a 10-word query takes less than 50 milliseconds on a standard CPU. The local Nool application loads the saved passage embeddings matrix and uses local CPU inference to encode user questions in real time.

---

## 6. Google Colab Workflow: Step-by-Step Guide

> [!IMPORTANT]
> The passage embeddings for the 50,000 HotpotQA sample must be generated in Google Colab by the user. They are not stored in the GitHub repository.

### Step 1: Open the Colab Notebook
Click the direct GitHub-to-Colab link:
[**Open Contriever Embeddings Notebook in Google Colab**](https://colab.research.google.com/github/nayanojwal0810/Nool/blob/main/notebooks/contriever_embeddings_colab.ipynb)

### Step 2: Enable GPU Hardware Acceleration
1. In the top navigation menu, click **Runtime** → **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** (or any available GPU).
3. Click **Save**.

### Step 3: Prepare Google Drive Folders & Upload Corpus
1. Open [Google Drive](https://drive.google.com).
2. Create the folder structure:
   `MyDrive/Nool/input/`
3. Upload your local prepared sample file:
   `Nool/data/processed/hotpotqa_sample/corpus.jsonl`
   to:
   `/content/drive/MyDrive/Nool/input/corpus.jsonl`

### Step 4: Run the Notebook
In Colab, click **Runtime** → **Run all** (or execute cells sequentially):
- **Cell 1**: Confirms CUDA GPU availability.
- **Cell 2**: Mounts Google Drive (`/content/drive`).
- **Cell 3**: Clones the Nool repository and records the current commit SHA.
- **Cell 4**: Installs `transformers==5.22.1` without modifying Colab's GPU PyTorch.
- **Cell 5**: Validates that `corpus.jsonl` has 50,000 passages and matches the expected SHA-256 (`c3970d91ebfd36d794066b42bd7acb85a744aabba2c97dd9a9bfde52d3dcbfbf`).
- **Cell 6**: Loads `facebook/contriever` at revision `2bd46a25019aeea091fd42d1f0fd4801675cf699` and runs a smoke test.
- **Cell 7**: Encodes all 50,000 passages in chunks of 2,500 with automatic checkpointing and CUDA out-of-memory recovery.
- **Cell 8**: Combines shards, verifies the 50,000 × 768 matrix, writes `index_manifest.json`, and packages `contriever_hotpotqa_sample.zip`.

### Checkpoint Resume Feature
If your Colab runtime times out or disconnects:
1. Reconnect to a GPU runtime.
2. Mount Google Drive.
3. Re-run Cell 7. The notebook checks `/content/drive/MyDrive/Nool/contriever_runs/hotpotqa_sample/checkpoints/` and automatically resumes from the latest completed shard.

---

## 7. Local Setup & Artifact Verification

### Step 1: Download & Extract the Artifact
1. From Google Drive, download:
   `/content/drive/MyDrive/Nool/contriever_runs/hotpotqa_sample/contriever_hotpotqa_sample.zip`
2. Extract the three files into your local project:
   ```
   Nool/indexes/contriever_hotpotqa_sample/
       ├── passage_embeddings.npy
       ├── document_ids.json
       └── index_manifest.json
   ```

### Step 2: Install Optional Dense Dependencies
In your local Windows virtual environment:
```powershell
.\.venv\Scripts\pip install -e ".[dense]"
```
This installs `torch==2.15.0` and `transformers==5.22.1` for local query encoding.

### Step 3: Run Artifact Verification
Verify that the downloaded files are uncorrupted and align with your local corpus:
```powershell
.\.venv\Scripts\python scripts/verify_contriever_artifact.py
```
This verifies SHA-256 checksums, float32 precision, dimension 768, and 1:1 ID alignment.

### Step 4: Run a Test Search
Query the dense retrieval index:
```powershell
.\.venv\Scripts\python scripts/search_contriever.py --query "Where was Marie Curie born?" --top-k 3
```

---

## 8. License & Attribution

Contriever is released by Meta AI under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

- **Paper**: Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, Edouard Grave. *Towards Unsupervised Dense Information Retrieval with Contrastive Learning*, arXiv:2112.09118, 2021.
- **Repository**: [https://github.com/facebookresearch/contriever](https://github.com/facebookresearch/contriever)
- **Model Card**: [https://huggingface.co/facebook/contriever](https://huggingface.co/facebook/contriever)
- **License**: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
