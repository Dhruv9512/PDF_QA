import os
from celery import Celery

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'PDF_QA.settings')

# Create the Celery app instance
app = Celery('PDF_QA')

# Set Redis as the broker and result backend (customizable via REDIS_URL env)
app.conf.broker_url = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
app.conf.result_backend = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')

# Load task modules from all registered Django app configs.
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

# Optional: Force import if tasks are not being picked up automatically
import chat_bot.email_tasks
