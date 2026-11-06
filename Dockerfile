FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl zstd \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst \
    -o /tmp/ollama-linux-amd64.tar.zst \
    && tar -C /usr -I zstd -xf /tmp/ollama-linux-amd64.tar.zst \
    && rm /tmp/ollama-linux-amd64.tar.zst

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd -m -u 1000 user \
    && chown -R user:user /app

USER user
ENV OLLAMA_MODELS=/home/user/.ollama/models
EXPOSE 8501
CMD ["bash", "scripts/start_web.sh"]
