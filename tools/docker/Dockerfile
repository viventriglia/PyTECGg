FROM python:3.13-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN uv pip install --system --no-cache-dir "pytecgg==1.0.2" "matplotlib>=3.10,<4"

COPY main.py .
COPY utils.py .

RUN mkdir -p /data/input /data/output /data/nav

ENTRYPOINT ["python", "main.py"]