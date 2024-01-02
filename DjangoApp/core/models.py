from django.db import models

# Create your models here.

class MeetFile(models.Model):
    file = models.FileField(upload_to='files/')
    transcript = models.TextField(default='')
    hindi_transcript = models.TextField(default='')
    marathi_transcript = models.TextField(default='')
    tamil_transcript = models.TextField(default='')
    kannada_transcript = models.TextField(default='')
    telgu_transcript = models.TextField(default='')
    summary = models.TextField(default='')
    sentiment_positive = models.FloatField(default=0.0)
    sentiment_negative = models.FloatField(default=0.0)
    sentiment_neutral = models.FloatField(default=0.0)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return self.file.name
