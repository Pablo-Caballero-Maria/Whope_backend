from django.apps import AppConfig


class DummyConfig(AppConfig):
    default_auto_field: str = 'django.db.models.BigAutoField'
    name: str = 'dummy'
