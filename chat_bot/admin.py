from django.contrib import admin
from .models import Ans_pdf

# Register your models here.
@admin.register(Ans_pdf)
class AnsPdfAdmin(admin.ModelAdmin):
    list_display = ('question', 'answer')
    search_fields = ('question', 'answer')