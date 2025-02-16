import json
from typing import Dict

from channels.generic.websocket import AsyncWebsocketConsumer
from utils.db_utils import generate_token

from whope.settings import PUBLIC_KEY_BYTES


class LoginConsumer(AsyncWebsocketConsumer):
    async def connect(self) -> None:
        await self.accept()
        # send the public key encoded in base64 (the PEM format is already b64) becuase the method in js to turn it into binary needs it to be in base64
        public_key_pem: str = PUBLIC_KEY_BYTES.decode("utf-8")
        public_key_clean: str = "".join(public_key_pem.strip().splitlines()[1:-1])
        await self.send(json.dumps({"public_key": public_key_clean}))

    async def disconnect(self, close_code: int) -> None:
        pass

    async def receive(self, text_data: str) -> None:
        data: Dict[str, str] = json.loads(text_data)
        result: Dict[str, str] = await generate_token(data)
        await self.send(json.dumps(result))
