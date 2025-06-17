from django.db import models

# Create your models here.
class Ans_pdf(models.Model):
    question = models.TextField()
    answer = models.TextField()

    def __str__(self):
        return self.question[:50]  