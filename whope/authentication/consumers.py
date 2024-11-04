import json
from typing import Any, Dict

from channels.generic.websocket import AsyncWebsocketConsumer
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from passlib.hash import bcrypt

from whope.settings import get_motor_db


class AuthenticationConsumer(AsyncWebsocketConsumer):
    async def connect(self) -> None:
        await self.accept()

    async def disconnect(self, close_code: int) -> None:
        pass

    async def receive(self, text_data: Dict[str, str]) -> None:
        data: Dict[str, str] = json.loads(text_data)
        action: str = data.get("action", "")

        if action == "register":
            await self.register_user(data)
        elif action == "login":
            await self.login_user(data)

    async def register_user(self, data: Dict[str, Any]) -> None:
        username: str = data.get("username", "")
        password: str = data.get("password", "")
        is_worker: bool = data.get("is_worker", False)
        status: str = "Disconnected"

        if not password or not username:
            await self.send(json.dumps({"error": "Username and password are required."}))
            return
        users: AsyncIOMotorCollection = await self.get_users()
        user: Dict[str, Any] = await users.find_one({"username": username})

        if user is not None:
            await self.send(json.dumps({"error": "User already exists."}))
            return

        hashed_password: str = bcrypt.hash(password)
        print("hola")
        await users.insert_one(
            {
                "username": username,
                "password": hashed_password,
                "is_worker": is_worker,
                "status": status,
            }
        )
        await self.send(json.dumps({"message": "User registered successfully."}))

    async def login_user(self, data: Dict[str, Any]) -> None:
        username: str = data.get("username", "")
        password: str = data.get("password", "")

        if not password or not username:
            await self.send(json.dumps({"error": "Username and password are required."}))
            return
        users: AsyncIOMotorCollection = await self.get_users()
        user: Dict[str, Any] = await users.find_one({"username": username})
        if not bcrypt.verify(password, user.get("password", None)):
            await self.send(json.dumps({"error": "Invalid password."}))
            return

        await self.set_user_status(user, "Free")

        from rest_framework_simplejwt.tokens import RefreshToken

        user_id: str = str(user.get("_id", None))
        refresh: RefreshToken = RefreshToken()
        refresh["user_id"] = user_id
        refresh["username"] = username
        refresh["is_worker"] = user.get("is_worker", False)
        tokens: Dict[str, str] = {
            "refresh": str(refresh),
            "access": str(refresh.access_token),
        }

        await self.send(json.dumps({"message": "Login successful.", "tokens": tokens}))

    async def set_user_status(self, user: Dict[str, Any], status: str) -> None:
        users: AsyncIOMotorCollection = await self.get_users()
        await users.update_one({"_id": user.get("user_id", None)}, {"$set": {"status": status}})

    async def get_users(self) -> AsyncIOMotorCollection:
        db: AsyncIOMotorDatabase = await get_motor_db()
        users: AsyncIOMotorCollection = db["users"]
        return users
