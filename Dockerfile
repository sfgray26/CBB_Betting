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

CMD ["/bin/sh", "-c", "if [ \"${RUN_STARTUP_MIGRATIONS:-true}\" = \"true\" ]; then python scripts/migrate_v9_live_data.py && python scripts/migrate_v10_user_preferences.py; else echo 'Skipping startup migrations'; fi && if [ \"${RUN_STARTUP_DB_INIT:-true}\" = \"true\" ]; then python -m backend.models; else echo 'Skipping startup DB init'; fi && uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
