from typing import List

from django.urls import path

from .consumers import RegisterConsumer

websocket_urlpatterns: List[path] = [
    path("ws/register/", RegisterConsumer.as_asgi()),
]
