"""
ASGI config for whope project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/asgi/
"""

import os

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from chat.routing import websocket_urlpatterns as chat_websocket_urlpatterns
from django.core.asgi import get_asgi_application
from login.routing import websocket_urlpatterns as login_websocket_urlpatterns
from register.routing import websocket_urlpatterns as register_websocket_urlpatterns

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "whope.settings")

application: "ASGIApplication" = ProtocolTypeRouter(
    {
        "http": get_asgi_application(),
        "websocket": AuthMiddlewareStack(URLRouter(register_websocket_urlpatterns + login_websocket_urlpatterns + chat_websocket_urlpatterns)),
    }
)
