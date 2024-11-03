from celery import Celery
from django.conf import settings

app = Celery("whope")

app.conf.broker_url = settings.RABBITMQ_URI
app.conf.result_backend = settings.CELERY_RESULT_BACKEND
app.conf.worker_pool = "prefork"
app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]
app.conf.broker_connection_retry_on_startup = True

app.autodiscover_tasks()
