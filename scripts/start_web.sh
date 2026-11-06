#!/usr/bin/env bash
set -euo pipefail

python -m nool.app.hosted_secrets
export OLLAMA_HOST=127.0.0.1:11434
ollama serve &
model_server=$!

for attempt in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null; then
    break
  fi
  if ! kill -0 "$model_server" 2>/dev/null; then
    exit 1
  fi
  sleep 1
done

curl -fsS http://127.0.0.1:11434/api/tags >/dev/null
ollama pull qwen3:4b-instruct

python -m streamlit run streamlit_app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --server.headless=true \
  --browser.gatherUsageStats=false &
website=$!

wait -n "$model_server" "$website"
