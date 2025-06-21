from typing import List

from django.urls import path

from .consumers import LoginConsumer

websocket_urlpatterns: List[path] = [
    path("ws/login/", LoginConsumer.as_asgi()),
]
