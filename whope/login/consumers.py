import json
from typing import Dict

from channels.generic.websocket import AsyncWebsocketConsumer
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from utils.crypto_utils import decrypt_with_private_key, encrypt_with_symmetric_key, decrypt_with_symmetric_key

from whope.settings import PRIVATE_KEY_BYTES, PUBLIC_KEY_BYTES, get_motor_db


class LoginConsumer(AsyncWebsocketConsumer):
    async def connect(self) -> None:
        await self.accept()
        # send the public key encoded in base64 (the PEM format is already b64) becuase the method in js to turn it into binary needs it to be in base64
        public_key_pem: str = PUBLIC_KEY_BYTES.decode("utf-8")
        public_key_clean: str = "".join(public_key_pem.strip().splitlines()[1:-1])
        await self.send(json.dumps({"public_key": public_key_clean}))

    async def disconnect(self, close_code: int) -> None:
        pass

    async def receive(self, text_data: Dict[str, str]) -> None:
        data: Dict[str, str] = json.loads(text_data)
        encrypted_username: str = data.get("encrypted_username", "")
        encrypted_password: str = data.get("encrypted_password", "")
        encrypted_symmetric_key: str = data.get("encrypted_symmetric_key", "")
        symmetric_key: bytes = decrypt_with_private_key(encrypted_symmetric_key.encode("utf-8"), PRIVATE_KEY_BYTES)

        users: AsyncIOMotorCollection = await self.get_users()
        user: Dict[str, str] = await users.find_one({"username": encrypted_username})

        if encrypted_password != user.get("password", None):
            await self.send(json.dumps({"error": "Invalid password."}))
            return

        from rest_framework_simplejwt.tokens import RefreshToken
        refresh: RefreshToken = RefreshToken()
        decrypted_username: str = decrypt_with_symmetric_key(encrypted_username.encode("utf-8"), symmetric_key)
        # no need to store the username encrypted inside the token since the token itself will be encrypted
        refresh["username"] = decrypted_username
        refresh["is_worker"] = user.get("is_worker", None)
        refresh["user_id"] = str(user.get("_id", None))
        access_token: str = str(refresh.access_token)
        tokens: Dict[str, str] = {
            "refresh": str(refresh),
            "access": encrypt_with_symmetric_key(access_token, symmetric_key),
        }

        await self.send(json.dumps({"message": "Login successful", "tokens": tokens}))

    async def get_users(self) -> AsyncIOMotorCollection:
        db: AsyncIOMotorDatabase = await get_motor_db()
        users: AsyncIOMotorCollection = db["users"]
        return users
