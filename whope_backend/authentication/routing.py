from typing import List

from django.urls import path

from .consumers import AuthenticationConsumer

websocket_urlpatterns: List[path] = [
    path("ws/authentication/", AuthenticationConsumer.as_asgi()),
]
