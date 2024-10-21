from django.urls import path
from .consumers import AuthenticationConsumer
from typing import List

websocket_urlpatterns: List[path] = [
    path('ws/authentication/', AuthenticationConsumer.as_asgi()),
]
