#!/bin/bash

echo "✅ Waiting for Redis to be ready..."
until nc -z redis 6379; do
  echo "⏳ Waiting for Redis..."
  sleep 1
done

echo "🚀 Starting Celery Worker"
source /opt/conda/etc/profile.d/conda.sh
conda activate myenv
celery -A PDF_QA worker --loglevel=info --pool=solo
