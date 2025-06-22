import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'PDF_QA.settings')

app = Celery('PDF_QA')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()  # This will autodiscover tasks in all installed apps

# Optionally, force import your tasks module if autodiscover doesn't work:
import chat_bot.email_tasks
