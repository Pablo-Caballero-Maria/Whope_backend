import json
from datetime import datetime, timezone
from typing import Any, Dict
from urllib.parse import parse_qs
from rest_framework_simplejwt.exceptions import TokenError
from aio_pika import Channel, IncomingMessage, Queue, RobustConnection, connect_robust
from asgiref.sync import sync_to_async
from bson import ObjectId
from celery.result import AsyncResult
from channels.generic.websocket import AsyncWebsocketConsumer
from chat.tasks import check_for_non_workers_task
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from utils.crypto_utils import decrypt_with_symmetric_key, encrypt_with_symmetric_key, decrypt_with_private_key 
from whope.settings import RABBITMQ_URI, get_motor_db, PUBLIC_KEY_BYTES, PRIVATE_KEY_BYTES
import base64

# sequence: receive (server receives encrypted symmetric key) -> initialize_connection (server decrypts symmetric key and token)
# -> assign_room (server assigns room) -> receive (server receives message) -> send_message (server sends message)
class ChatConsumer(AsyncWebsocketConsumer):

    async def connect(self) -> None:
        # this is just initialization (it must be defined):
        self.room_name: str = None
        self.user: Dict[str, str] = None
        await self.accept()

    async def initialize_connection(self, data: Dict[str, str]) -> None:
        encrypted_symmetric_key: str = data.get("encrypted_symmetric_key", None)
        encrypted_token: str = data.get("encrypted_token", None)
        self.symmetric_key: bytes = decrypt_with_private_key(encrypted_symmetric_key.encode("utf-8"), PRIVATE_KEY_BYTES)
        self.token: str = decrypt_with_symmetric_key(encrypted_token.encode("utf-8"), self.symmetric_key)
        self.user: Dict[str, str] = await self.get_user_from_token(self.token)
        # channel_name is automatically set and it needs to be saved so than the worker can add the non_worker to the room
        # once he finds one
        await self.save_channel_name(self.user, self.channel_name)
        # users have 3 status: Free (non worker chatting with ai/worker waiting), Busy (worker chatting with non worker or viceversa),
        # Disconnected (not connected to any ws)
        await self.set_user_status(self.user, "Free")
        await self.assign_room()

    async def disconnect(self, close_code: int) -> None:
        if self.user and self.room_name:
            AsyncResult(self.celery_task_id).revoke(terminate=True) if self.user.get("is_worker", None) == "True" else None
            self.rabbit_connection.close() if hasattr(self, "rabbit_connection") else None
            await self.channel_layer.group_discard(self.room_name, self.user.get("channel_name", None))
            await self.save_channel_name(self.user, None)
            await self.set_user_status(self.user, "Disconnected")
            # this message will only be received by the OTHER user
            await self.channel_layer.group_send(self.room_name, {"type": "user_disconnection"})

    async def send_message(self, data: Dict[str, str]) -> None:
        message: str = data.get("message", None)
        username: str = data.get("username", None)
        await self.preprocess_message(message, self.user)
        await self.save_message(message)

        await self.channel_layer.group_send(
            self.room_name,
            {
                "type": "chat_message",
                "username": username,
                "message": message,
            },
        )

        if "virtual_room" in self.room_name:
            # TODO: use ai bot here
            decrypted_message: str = decrypt_with_symmetric_key(message.encode("utf-8"), self.symmetric_key)
            answer: str = f"AI bot says: your message { decrypted_message } has been received."
            encrypted_answer: str = encrypt_with_symmetric_key(answer, self.symmetric_key)
            await self.channel_layer.group_send(
                self.room_name,
                {
                    "type": "chat_message",
                    "username": encrypt_with_symmetric_key("AI bot", self.symmetric_key),
                    "message": encrypted_answer,
                },
            )

    async def receive(self, text_data: str) -> None:
        data: Dict[str, str] = json.loads(text_data)
        action: str = data.get("action", None)

        if action == "send_message":
            await self.send_message(data)
        elif action == "initialize_connection":
            await self.initialize_connection(data)

    async def chat_message(self, event: Dict[str, str]) -> None:
        message: str = event.get("message", None)
        await self.send(
            text_data=json.dumps(
                {
                    "username": event.get("username", None),
                    "message": message,
                }
            )
        )

    async def save_message(self, message_content: str) -> None:
        message: Dict[str, str] = {
            "message": message_content,
            "created_at": datetime.now(timezone.utc),
        }
        users: AsyncIOMotorCollection = await self.get_users()
        await users.update_one(
            {"_id": ObjectId(self.user.get("user_id", None))},
            {"$push": {"messages": message}},
        )

    async def get_users(self) -> AsyncIOMotorCollection:
        db: AsyncIOMotorDatabase = await get_motor_db()
        users: AsyncIOMotorCollection = db["users"]
        return users

    async def get_user_from_token(self, token: str) -> Dict[str, str]:
        from rest_framework_simplejwt.tokens import UntypedToken
        decoded_token: UntypedToken = UntypedToken(token)
        user_id: str = decoded_token.get("user_id", None)
        username: str = decoded_token.get("username", None)
        is_worker: bool = decoded_token.get("is_worker", None)
        return {"user_id": user_id, "username": username, "is_worker": is_worker}

    async def assign_room(self) -> None:
        if self.user.get("is_worker", None) == "True":
            self.room_name: str = f"{self.user.get('username', None)}_waiting_room"
            await self.listen_to_rabbit()
            celery_task: AsyncResult = await sync_to_async(check_for_non_workers_task.delay)(self.user.get("username", None))
            self.celery_task_id: str = celery_task.id
        else:
            self.room_name: str = f"{self.user.get('username', None)}_virtual_room"

        await self.channel_layer.group_add(self.room_name, self.user.get("channel_name", None))

    async def process_non_worker_assignment(self, worker_username, free_non_worker):
        AsyncResult(self.celery_task_id).revoke(terminate=True)
        free_non_worker_username: str = decrypt_with_symmetric_key(free_non_worker.get("username", "").encode("utf-8"), self.symmetric_key)
        new_room_name: str = f"{worker_username}_{free_non_worker_username}_real_room"
        self.room_name: str = new_room_name
        await self.channel_layer.group_send(
            f"{free_non_worker_username}_virtual_room",
                    {"type": "update_room_name", "new_room_name": new_room_name, "new_symmetric_key": self.symmetric_key},
        )
        await self.channel_layer.group_discard(
            f"{free_non_worker_username}_virtual_room",
            free_non_worker.get("channel_name", None),
        )
        await self.set_user_status(self.user, "Busy")
        await self.set_user_status(free_non_worker, "Busy")
        await self.channel_layer.group_add(new_room_name, self.user.get("channel_name", None))
        await self.channel_layer.group_add(new_room_name, free_non_worker.get("channel_name", None))

    async def set_user_status(self, user: Dict[str, str], status: str) -> None:
        users: AsyncIOMotorCollection = await self.get_users()
        await users.update_one({"_id": ObjectId(user.get("user_id", None))}, {"$set": {"status": status}})

    async def save_channel_name(self, user: Dict[str, str], channel_name: str) -> None:
        users: AsyncIOMotorCollection = await self.get_users()
        await users.update_one({"_id": ObjectId(user.get("user_id", None))}, {"$set": {"channel_name": channel_name}})
        # so that locally its also updated without having to call db again
        self.user["channel_name"]: str = channel_name

    async def update_room_name(self, event: Dict[str, Any]) -> None:
        self.room_name: str = event.get("new_room_name", None)
        # client sends message with his symmetric key and the server do a translation to decrypt it and then
        # encrypt it with the new symmetric key and send it to the worker (to the non worker is sent with the old symmetric key)
        self.new_symmetric_key: bytes = event.get("new_symmetric_key", None)
        new_symmetric_key_str = base64.b64encode(self.new_symmetric_key).decode('utf-8')
        # send new symmetric key to non worker
        await self.send(
            text_data=json.dumps(
                {
                    "new_symmetric_key": encrypt_with_symmetric_key(new_symmetric_key_str, self.symmetric_key),
                }
            )
        )

    async def preprocess_message(self, message: str, user: Dict[str, str]):
        await self.send(text_data=json.dumps({"error": "Message must not be empty."})) if not message else None
        # if theres an error with the token and i have to send the error response, i cannot do it before here (in get_user_from_token or in connect)
        # cuz the channel isnt even open
        await self.send(text_data=json.dumps(user)) if "error" in user else None
        # TODO: maybe add more checks

    async def user_disconnection(self, event: Dict[str, str]) -> None:
        if self.user.get("is_worker", None) == "True":
            # if the non worker disconnected, the worker goes back to pinging
            await self.set_user_status(self.user, "Free")
            await self.assign_room()
        else:
            # if the worker disconnected, the non worker closes the connection
            await self.set_user_status(self.user, "Disconnected")
            await self.close()

    async def listen_to_rabbit(self) -> None:
        connection: RobustConnection = await connect_robust(RABBITMQ_URI)
        self.rabbit_connection: RobustConnection = connection
        channel: Channel = await self.rabbit_connection.channel()
        await channel.set_qos(prefetch_count=1)
        queue: Queue = await channel.declare_queue(f"{self.user.get('username', None)}_assignment_queue")

        async def on_message(message: IncomingMessage) -> None:
            async with message.process():
                data: Dict[str, str] = json.loads(message.body)
                worker_username: str = data.get("worker_username", None)
                free_non_worker: str = data.get("free_non_worker", None)

                if worker_username == self.user.get("username"):
                    await self.process_non_worker_assignment(worker_username, free_non_worker)

        await queue.consume(on_message)
