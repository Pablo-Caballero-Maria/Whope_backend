import os
from typing import Dict

from channels.testing import WebsocketCommunicator
from django.test import TransactionTestCase
from motor.motor_asyncio import AsyncIOMotorDatabase

from whope_backend.asgi import application
from whope_backend.settings import get_motor_db

os.environ["TESTING"] = "True"


class TestChatConsumer(TransactionTestCase):

    async def websocket_connect(
        self, room_name: str, token: str = ""
    ) -> WebsocketCommunicator:
        communicator: WebsocketCommunicator = WebsocketCommunicator(
            application,
            f"/ws/chat/{room_name}/",
            headers=[
                (b"Authorization", f"Bearer {token}".encode("utf-8")),
            ],
        )

        connected, _ = await communicator.connect()
        assert connected

        db: AsyncIOMotorDatabase = await get_motor_db()
        await db.drop_collection("messages")

        return communicator

    async def websocket_authentication_connect(self) -> WebsocketCommunicator:
        communicator: WebsocketCommunicator = WebsocketCommunicator(
            application, "/ws/authentication/"
        )
        connected, _ = await communicator.connect()
        assert connected

        db: AsyncIOMotorDatabase = await get_motor_db()
        await db.drop_collection("users")

        return communicator

    async def register_and_login_user(self) -> str:
        auth_communicator: WebsocketCommunicator = (
            await self.websocket_authentication_connect()
        )

        register_data: Dict[str, str] = {
            "action": "register",
            "username": "username_test",
            "password": "password_test",
        }
        await auth_communicator.send_json_to(register_data)
        await auth_communicator.receive_json_from()

        login_data: Dict[str, str] = {
            "action": "login",
            "username": "username_test",
            "password": "password_test",
        }
        await auth_communicator.send_json_to(login_data)
        response: Dict[str, str] = await auth_communicator.receive_json_from()
        assert response.get("message", None) == "Login successful."
        assert "tokens" in response

        access_token: str = response.get("tokens", None).get("access", None)
        await auth_communicator.disconnect()
        return access_token

    async def test_send_and_receive_message(self) -> None:
        token: str = await self.register_and_login_user()

        communicator: WebsocketCommunicator = await self.websocket_connect(
            "testroom", token
        )

        message_data: Dict[str, str] = {
            "message": "Hello, world!",
        }
        await communicator.send_json_to(message_data)

        response: Dict[str, str] = await communicator.receive_json_from()
        assert response.get("username", None) == "username_test"
        assert response.get("message", None) == "Hello, world!"

        await communicator.disconnect()

    async def test_missing_message(self) -> None:
        token: str = await self.register_and_login_user()
        communicator: WebsocketCommunicator = await self.websocket_connect(
            "testroom", token
        )
        message_data = {}
        await communicator.send_json_to(message_data)

        response = await communicator.receive_json_from()
        assert response.get("error", None) == "Message must not be empty."

        await communicator.disconnect()

    async def test_missing_token(self) -> None:
        communicator: WebsocketCommunicator = await self.websocket_connect("testroom")
        message_data: Dict[str, str] = {
            "message": "Hello, world!",
        }
        await communicator.send_json_to(message_data)
        response: Dict[str, str] = await communicator.receive_json_from()
        assert response.get("error", None) == "Token must not be empty."

        await communicator.disconnect()

    async def test_invalid_token(self) -> None:
        communicator: WebsocketCommunicator = await self.websocket_connect(
            "testroom", "invalid_token"
        )
        message_data: Dict[str, str] = {
            "message": "Hello, world!",
        }
        await communicator.send_json_to(message_data)
        response: Dict[str, str] = await communicator.receive_json_from()
        assert response.get("error", None) == "Invalid token."

        await communicator.disconnect()
