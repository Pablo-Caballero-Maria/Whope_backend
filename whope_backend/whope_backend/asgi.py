"""
ASGI config for whope_backend project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import hello_world.routing
import authentication.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'whope_backend.settings')

application: "ASGIApplication" = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            hello_world.routing.websocket_urlpatterns + 
            authentication.routing.websocket_urlpatterns
        )
    ),
})

