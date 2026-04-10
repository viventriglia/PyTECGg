FROM python:3.13-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN uv pip install --system --no-cache-dir "pytecgg==1.2.0" "matplotlib>=3.10,<4" "plotly>=6.2,<6.3"

COPY main.py .
COPY utils.py .

RUN mkdir -p /data/input /data/output /data/nav

ENTRYPOINT ["python", "main.py"]