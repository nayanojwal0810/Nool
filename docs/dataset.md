# Nool — Enterprise Knowledge Assistant: Retrieval Dataset

Nool uses the BEIR version of HotpotQA to measure how well its search methods find useful passages. The dataset provides questions and relevance records that identify passages known to help answer them. We compare BM25, Contriever, and hybrid search against those records. HotpotQA cannot tell us whether the future company website answers correctly or protects restricted documents.

## Source and fixed sample

The source archive contains 5,233,329 passages and 97,852 questions. Its recorded SHA-256 is `c62459dbc91d31329584507b1fa564b3b1abcee58800967113054ee0f5a15c62`. The archive audit found no missing question or passage references in its relevance records.

The working benchmark uses one reproducible sample:

| Item | Count |
|---|---:|
| Passages | 50,000 |
| Development questions | 500 |
| Test questions | 500 |
| Relevant passages retained | 1,981 |
| Background passages | 48,019 |

Every relevant passage for the selected questions is in the sample. Development and test question IDs do not overlap. The sample uses random seed 42, and its corpus SHA-256 is `c3970d91ebfd36d794066b42bd7acb85a744aabba2c97dd9a9bfde52d3dcbfbf`. The manifests in `data/manifests/` record the source and sample details.

The archive and processed passages are excluded from Git. Place the downloaded `hotpotqa.zip` in `data/raw/`. Run `scripts/audit_hotpotqa.py` to inspect the archive and `scripts/prepare_hotpotqa_sample.py` to recreate the sample. Both scripts run from the project root after installing the project. The scripts create their output directories when needed.

Results on these 50,000 passages cannot be compared directly with results on the full 5.23-million-passage collection. Some useful passages may lack relevance labels, so reported scores measure matches to the known labels.

The source dataset is [BEIR HotpotQA](https://huggingface.co/datasets/BeIR/hotpotqa), listed under CC BY-SA 4.0. Review its terms before redistributing data.
