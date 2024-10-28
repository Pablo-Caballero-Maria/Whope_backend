from django.urls import path, re_path
from .consumers import ChatConsumer
from typing import List

websocket_urlpatterns: List[path] = [
    path('ws/chat/', ChatConsumer.as_asgi()),  # Solo necesitas el room_name
]
