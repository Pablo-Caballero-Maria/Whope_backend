from django.urls import path
from . import consumers
from typing import List

websocket_urlpatterns: List[path] = [
    path("ws/hello_world", consumers.HelloWorldConsumer.as_asgi()),  
]
