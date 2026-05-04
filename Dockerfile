FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install core requirements
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly install performance-critical async packages
# (Railway's bulk install sometimes times out silently)
RUN pip install --no-cache-dir asyncpg msgpack

COPY . .

EXPOSE 8000

CMD ["/bin/sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
