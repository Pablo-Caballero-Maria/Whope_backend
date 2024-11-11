import json
from typing import Dict

from channels.generic.websocket import AsyncWebsocketConsumer
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase

from whope.settings import get_motor_db


class RegisterConsumer(AsyncWebsocketConsumer):
    async def connect(self) -> None:
        await self.accept()

    async def disconnect(self, close_code: int) -> None:
        pass

    async def receive(self, text_data: Dict[str, str]) -> None:
        data: Dict[str, str] = json.loads(text_data)

        encrypted_username: str = data.get("username", "")
        encrypted_password: str = data.get("password", "")
        is_worker: str = data.get("is_worker", "False")
        status: str = "Disconnected"

        users: AsyncIOMotorCollection = await self.get_users()
        user: Dict[str, str] = await users.find_one({"username": encrypted_username})

        if user is not None:
            await self.send(json.dumps({"error": "User already exists."}))
            return

        await users.insert_one(
            {
                "username": encrypted_username,
                "password": encrypted_password,
                "is_worker": is_worker,
                "status": status,
            }
        )
        await self.send(json.dumps({"message": "User registered successfully."}))

    async def get_users(self) -> AsyncIOMotorCollection:
        db: AsyncIOMotorDatabase = await get_motor_db()
        users: AsyncIOMotorCollection = db["users"]
        return users
