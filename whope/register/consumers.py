import json
from typing import Dict

from channels.generic.websocket import AsyncWebsocketConsumer
from utils.db_utils import store_user


class RegisterConsumer(AsyncWebsocketConsumer):
    async def connect(self) -> None:
        await self.accept()

    async def disconnect(self, close_code: int) -> None:
        pass

    async def receive(self, text_data: str) -> None:
        data: Dict[str, str] = json.loads(text_data)
        result: Dict[str, str] = await store_user(data)
        await self.send(json.dumps(result))
