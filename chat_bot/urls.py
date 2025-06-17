from django.urls import path
from .views import pdf

urlpatterns = [
    path("pdf/",pdf.as_view(), name="pdf"),
]