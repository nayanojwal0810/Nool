# Nool — Enterprise Knowledge Assistant: Company Answer Evaluation

## What we checked

We ran Qwen3 4B Instruct on the original 45 fictional company documents. The 162 initial-state cases contain 150 questions with expected sources, seven questions aimed at restricted documents, and five questions the documents cannot answer. Hybrid search supplied up to five permitted documents. The answer path selected relevant passages and checked each citation against current source text. Every case used a 600-second model timeout.

The complete predictions are saved locally under `outputs/`. The 50,000-passage search benchmark and company question set were unchanged.

## Results

| Question type | Cases | Cited every expected source |
|---|---:|---:|
| Direct | 45 | 41 |
| Reworded | 40 | 34 |
| Two-source | 40 | 3 |
| Near miss | 25 | 24 |
| Total answerable | 150 | 102 |

Across all 162 cases, Nool returned 113 answers, 18 unsupported responses, and 31 unavailable responses. Of the unavailable responses, 28 were model timeouts: 25 two-source and three reworded questions. Three were invalid model output.

Nool refused all five questions without a document answer and six of seven restricted-target questions. In the remaining restricted-target case, it gave an irrelevant answer from a permitted document. It did not cite the restricted target document. The separate permission-change and document-update cases are tested in code, outside this initial-state run.

## Interpretation

Hybrid search found every expected document for 142 of the 150 answerable questions. The gap between 142 found and 102 cited shows that finding the documents does not yet produce a complete answer. Two-source questions are the largest weakness, including both timeouts and answers that cite only one source.

An earlier Qwen 2.5 3B run cited every expected source in 86 of 150 answerable cases. It used a different passage selector and mixed timeouts, so these runs do not isolate the effect of changing models. Citation coverage also does not prove that every answer is factually correct. A full human review of the answer text remains to be done.

These results were recorded before the current separate-part answer path. That path checks two clear requests separately, but it has not yet been measured on the full question set. We will review fresh questions and rerun the complete set before reporting any improvement. Answers that address a related policy instead of the user's question remain a risk to check manually.
