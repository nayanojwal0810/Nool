# Nool — Enterprise Knowledge Assistant: Answer Guide

Nool's answer path takes the first five current documents that the selected demo user may access. It chooses the most relevant passage from each document and may include a second passage from the top two documents. It sends at most seven passages to a local Qwen model through Ollama. A supported answer is short and carries source citations with an exact excerpt from each cited passage. A one-character citation ID typo is corrected only when its exact quote uniquely matches a permitted passage. The answer path checks document status, version, and access both before and after the model call, so a change during generation cannot release the old answer.

When the passages do not support the question, Nool responds: “I couldn't verify an answer from the documents available to you. Please check with the relevant team before acting on it.” A partial answer states only the verified part and makes clear that the response is limited to what the documents confirm. The response does not mention restricted documents.

## Run an answer

The company index and local SQLite database must match. For an optional local check, [install Ollama for Windows](https://ollama.com/download/windows), then download [Qwen3 4B Instruct](https://ollama.com/library/qwen3:4b-instruct) with `ollama pull qwen3:4b-instruct`. Keep Ollama running and run:

```powershell
.\.venv\Scripts\python scripts/answer_company.py "How many annual leave days do I get?" --user employee_demo --offline
```

The model endpoint is limited to the local computer. The `--model` option selects a different installed Ollama model. Add `--cpu-only` to keep Ollama inference on the CPU. If the model is unavailable, Nool returns a neutral temporary-error message instead of inventing an answer.

## Review model outputs

Run `scripts/evaluate_company_answers.py --limit 5 --cpu-only --offline` for a short local check, or omit `--limit` for all 162 initial-state cases. Use `--case-ids Q066,Q131,Q093` to check named cases before a full run. Add `--resume` to continue an interrupted run with the same model, CPU setting, and question file. The script records answers, citations, retrieved documents, chosen passages, and model failure reasons. The [measured results](company_answer_evaluation.md) report the earlier complete Qwen3 CPU run and its limits. Review factual accuracy and whether every claim is supported by its cited excerpt; automatic counts are not an answer-accuracy score.

The four document-update and four permission-change cases need state changes and are tested separately. The 150-question retrieval result measures source finding, not the model's answer quality. An exact excerpt proves where a citation came from; it does not by itself prove that every statement in a generated answer follows from that excerpt. The demo user selector is not production authentication.
