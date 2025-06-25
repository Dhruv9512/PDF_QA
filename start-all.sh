#!/bin/bash

echo "✅ Waiting for Redis..."
# Extract hostname from CELERY_BROKER_URL (e.g., redis://redis:6379/0)
REDIS_HOST=$(echo "$CELERY_BROKER_URL" | cut -d@ -f2 | cut -d: -f1)

until nc -z "$REDIS_HOST" 6379; do
  echo "⏳ Redis ($REDIS_HOST) not ready yet..."
  sleep 1
done

echo "🚀 Starting Celery Worker in background"
source /opt/conda/etc/profile.d/conda.sh
conda activate myenv

celery -A PDF_QA worker --loglevel=info --pool=solo &

echo "🚀 Starting Gunicorn Web Server"
exec gunicorn PDF_QA.wsgi --workers 3 --worker-class gevent --timeout 120 --bind 0.0.0.0:8000
