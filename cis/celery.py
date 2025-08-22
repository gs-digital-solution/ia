import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cis.settings')  # adapte si projet != cis
app = Celery('cis')

app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()