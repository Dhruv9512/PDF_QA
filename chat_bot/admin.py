from django.contrib import admin
from .models import ReferalPDF

# Register your models here.
@admin.register(ReferalPDF)
class ReferalPDFAdmin(admin.ModelAdmin):
    list_display = ('pdf_id',)
  