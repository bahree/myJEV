FROM python:3.12.3-slim-bookworm@sha256:afc139a0a640942491ec481ad8dda10f2c5b753f5c969393b12480155fe15a63
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/cache/huggingface
WORKDIR /app
COPY requirements.lock pyproject.toml LICENSE ./
RUN pip install --no-cache-dir -r requirements.lock
# Triton compiles a small host launcher during the first real GPU forward.
RUN apt-get update && apt-get install --no-install-recommends -y gcc libc6-dev && rm -rf /var/lib/apt/lists/*
COPY src ./src
RUN pip install --no-deps .
EXPOSE 8000
HEALTHCHECK --interval=30s --start-period=180s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz')"
CMD ["uvicorn", "myjev.server:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--workers", "1"]
