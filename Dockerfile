FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Copy dependency specs first (cache layer)
COPY pyproject.toml ./
COPY packages/ packages/
COPY experiments/ experiments/

# Install the project (exp extras for full functionality)
RUN pip install --no-cache-dir -e ".[exp]"

# Copy remaining source
COPY . .

# Render injects PORT; default to 8000 locally
ENV PORT=8000
EXPOSE 8000

CMD uvicorn ads_api.main:app --host 0.0.0.0 --port $PORT
