# Nool — Enterprise Knowledge Assistant: Answer Guide

Nool's answer path starts with up to five current documents that the selected demo user may access. It searches each clear part of a compound question and, when useful, the final request without its introductory situation. These extra searches can add one document without removing the original results. Nool chooses relevant passages for Qwen through Ollama. It includes relevant procedures for action questions, approval policies for approval questions, and exception rules for questions with amounts. For a compound question, the model selects source IDs for each part, and Nool builds the displayed answer from the matching source wording. It rejects a part when its cited text has no meaningful term in common with that request. It also rejects unknown IDs and checks the current permitted passages before copying citation text. For a single-part answer, it rejects a stated day-count deadline when its starting event differs from the cited rule. Document status, version, and access are checked both before and after the model call, so a change during generation cannot release the old answer.

When the passages do not support the question, Nool responds: “I couldn't verify an answer from the documents available to you. Please check with the relevant team before acting on it.” A partial answer states only the verified part and makes clear that the response is limited to what the documents confirm. The response does not mention restricted documents.

## Run an answer

The company index and local SQLite database must match. For an optional local check, [install Ollama for Windows](https://ollama.com/download/windows), then download [Qwen3 4B Instruct](https://ollama.com/library/qwen3:4b-instruct) with `ollama pull qwen3:4b-instruct`. Keep Ollama running and run:

```powershell
.\.venv\Scripts\python scripts/answer_company.py "How many annual leave days do I get?" --user employee_demo --offline
```

The model endpoint is limited to the local computer. The `--model` option selects a different installed Ollama model. If the model is unavailable, Nool returns a neutral temporary-error message instead of inventing an answer.

## Review model outputs

Run `scripts/evaluate_company_answers.py --limit 5 --offline` for a short local check, or omit `--limit` for all 162 initial-state cases. Use `--case-ids Q066,Q131,Q093` to check named cases before a full run. Add `--resume` to continue an interrupted run with the same model and question file. The script records answers, citations, retrieved documents, chosen passages, and model failure reasons. The [measured results](company_answer_evaluation.md) report the earlier complete Qwen3 run and its limits. Review factual accuracy and whether every claim is supported by its cited excerpt; automatic counts are not an answer-accuracy score.

The four document-update and four permission-change cases need state changes and are tested separately. The 150-question retrieval result measures source finding, not the model's answer quality. The separate-part check applies to clear two-part requests; it cannot identify every implied need for multiple documents. An exact excerpt proves where a citation came from; it does not by itself prove that every statement in a generated answer follows from that excerpt. The demo user selector is not production authentication.
