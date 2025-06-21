from typing import List

from django.urls import path

from .consumers import ChatConsumer

websocket_urlpatterns: List[path] = [
    path("ws/chat/", ChatConsumer.as_asgi()),  # Solo necesitas el room_name
]
