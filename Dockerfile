FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["/bin/sh", "-c", "python scripts/migrate_v9_live_data.py && python scripts/migrate_v10_user_preferences.py && python -m backend.models && uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
