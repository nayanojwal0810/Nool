# Nool — Enterprise Knowledge Assistant: Answer Guide

Nool's answer path takes the first five current documents that the selected demo user may access. It sends the found passage and each document's policy passage, when present, to a local Qwen model through Ollama. A supported answer is short and carries source citations with an exact excerpt from each cited passage. The answer path checks document status, version, and access both before and after the model call, so a change during generation cannot release the old answer.

When the passages do not support the question, Nool responds: “I couldn't verify an answer from the documents available to you. Please check with the relevant team before acting on it.” A partial answer states only the verified part and makes clear that the response is limited to what the documents confirm. The response does not mention restricted documents.

## Run an answer

The company index and local SQLite database must match. [Install Ollama for Windows](https://ollama.com/download/windows), then download the [Qwen 2.5 3B model](https://ollama.com/library/qwen2.5:3b) with `ollama pull qwen2.5:3b`. Keep Ollama running and run:

```powershell
.\.venv\Scripts\python scripts/answer_company.py "How many annual leave days do I get?" --user employee_demo --offline
```

The model endpoint is limited to the local computer. The `--model` option selects a different installed Ollama model. If the model is unavailable, Nool returns a neutral temporary-error message instead of inventing an answer.

## Review model outputs

Run `scripts/evaluate_company_answers.py --limit 5 --offline` for a short check, or omit `--limit` for all 162 initial-state cases. Add `--resume` to continue an interrupted run from its saved file. The separate [Colab answer notebook](../notebooks/company_answers_colab.ipynb) can run the same check with a GPU. The script saves each answer beside the expected answer and source IDs in `outputs/company_answer_predictions.jsonl`. The [measured results](company_answer_evaluation.md) report expected-source citation coverage and refusals. Review factual accuracy and whether every claim is supported by its cited excerpt; these automatic counts are not an answer-accuracy score.

The four document-update and four permission-change cases need state changes and are tested separately. The 150-question retrieval result measures source finding, not the model's answer quality. An exact excerpt proves where a citation came from; it does not by itself prove that every statement in a generated answer follows from that excerpt. The demo user selector is not production authentication.
