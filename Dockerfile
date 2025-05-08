FROM python:3.10-slim

WORKDIR /app

# Install git to allow transformers to download model card metadata (optional but common)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command prints embeddings for two sample sentences
ENTRYPOINT ["python"]
CMD ["embed.py"] 