from django.urls import path
from .consumers import ChatConsumer
from typing import List

websocket_urlpatterns: List[path] = [
    path('ws/chat/<str:room_name>/', ChatConsumer.as_asgi()),
]
