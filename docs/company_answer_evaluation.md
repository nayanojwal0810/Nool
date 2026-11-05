# Nool — Enterprise Knowledge Assistant: Company Answer Evaluation

## What we checked

We ran the local Qwen 2.5 3B model through Nool's answer path on the original 45 fictional company documents. The check used 150 questions with known sources, seven questions aimed at restricted documents, and five questions the documents cannot answer. Hybrid search supplied up to five permitted documents. The answer path also supplied each found document's policy passage when available and checked citations against exact source text.

We saved all 162 responses in `outputs/company_answer_predictions.jsonl`. The run started with a 240-second model timeout and resumed after interruption with a 600-second timeout. The counts below describe this run; a future comparison should use one timeout for every case.

## Results

| Question type | Cases | Cited every expected source |
|---|---:|---:|
| Direct | 45 | 36 |
| Reworded | 40 | 27 |
| Two-source | 40 | 6 |
| Near miss | 25 | 17 |
| Total answerable | 150 | 86 |

The model returned 54 answered, 59 partial, 36 unsupported, and 13 unavailable responses across all 162 cases. It refused all five questions with no document answer. It refused four of seven restricted-target questions. On the other three, it gave related information from permitted documents; it did not cite the restricted target documents. Those three responses did not meet the intended refusal behavior.

The 13 unavailable responses included 11 two-source questions and two reworded questions. The current output does not record whether each unavailable response came from a timeout, a connection error, or invalid model output.

## What the results mean

The [company search evaluation](company_search_evaluation.md) found every expected document for 142 of 150 answerable questions. This answer check is stricter: the model must use and cite those documents in its response. The gap is largest on questions needing two documents. Some clear single-document questions were also refused even when the correct policy passage was available.

The citation check confirms that each quoted excerpt appears in a current permitted passage. It does not prove that the full answer follows from that excerpt. We have not assigned a factual-accuracy score to the generated text. The demo users are test identities, not website authentication.

The answer path now chooses passages based on the question and records why an answer was unavailable or refused. We checked passage selection on all 150 answerable cases: it kept the expected documents in all 142 cases where hybrid search found them. It selected 783 passages in total, or 5.2 per question on average, with no more than seven for any question. This check measures source retention, not answer quality.

The full comparison of Qwen3 4B Instruct and Qwen 2.5 3B has not completed. It will use the same 162 cases, passage selector, and timeout for both models, including two-source answers and restricted-target refusals. We will report an improvement only after checking the saved answers. The 50,000-passage search benchmark and question set remain fixed.
