from django.db import models

# Create your models here.
class ReferalPDF(models.Model):
    pdf_id = models.CharField(max_length=150)
    def __str__(self):
        return self.pdf_id  