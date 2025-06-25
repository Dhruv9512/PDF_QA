import os
from django.core.wsgi import get_wsgi_application
from whitenoise import WhiteNoise
from pathlib import Path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "PDF_QA.settings")

BASE_DIR = Path(__file__).resolve().parent.parent  # Add this if not already present

application = get_wsgi_application()
application = WhiteNoise(application, root=str(BASE_DIR / "staticfiles"))
