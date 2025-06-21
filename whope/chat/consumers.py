import json
from typing import Any, Dict, List

from aio_pika import Channel, IncomingMessage, Queue, RobustConnection, connect_robust
from asgiref.sync import sync_to_async
from celery.result import AsyncResult
from channels.generic.websocket import AsyncWebsocketConsumer
from chat.tasks import check_for_non_workers_task
from utils.crypto_utils import (
    decrypt_with_private_key,
    decrypt_with_symmetric_key,
    encrypt_with_symmetric_key,
)
from utils.db_utils import (
    get_all_messages_from_user,
    get_user_from_token,
    save_channel_name,
    save_message,
    set_user_status,
)
from utils.management import get_enriched_message
from utils.nlp_utils import (
    generate_answer_from_enriched_message,
)

from whope.settings import PRIVATE_KEY_BYTES, RABBITMQ_URI


# sequence: receive (server receives encrypted symmetric key) -> initialize_connection (server decrypts symmetric key and token)
# -> assign_room (server assigns room) -> receive (server receives message) -> send_message (server sends message)
class ChatConsumer(AsyncWebsocketConsumer):

    async def connect(self) -> None:
        # this is just initialization (it must be defined):
        self.room_name: str = None
        self.user: Dict[str, Any] = None
        await self.accept()

    async def initialize_connection(self, data: Dict[str, str]) -> None:
        encrypted_symmetric_key: str = data.get("encrypted_symmetric_key", None)
        encrypted_token: str = data.get("encrypted_token", None)
        self.symmetric_key: bytes = decrypt_with_private_key(encrypted_symmetric_key.encode("utf-8"), PRIVATE_KEY_BYTES)
        self.token: str = decrypt_with_symmetric_key(encrypted_token, self.symmetric_key)
        self.user: Dict[str, Any] = await get_user_from_token(self.token)
        # channel_name is automatically set and it needs to be saved so than the worker can add the non_worker to the room
        # once he finds one
        await save_channel_name(self.user, self.channel_name)
        self.user["channel_name"] = self.channel_name
        # users have 3 status: Free (non worker chatting with ai/worker waiting), Busy (worker chatting with non worker or viceversa),
        # Disconnected (not connected to any ws)
        await set_user_status(self.user, "Free")
        await self.assign_room()

    async def disconnect(self, close_code: int) -> None:
        if self.user and self.room_name:
            AsyncResult(self.celery_task_id).revoke(terminate=True) if self.user.get("is_worker", None) == "True" else None
            self.rabbit_connection.close() if hasattr(self, "rabbit_connection") else None
            await self.channel_layer.group_discard(self.room_name, self.user.get("channel_name", None))
            await save_channel_name(self.user, None)
            await set_user_status(self.user, "Disconnected")
            # this message will only be received by the OTHER user
            await self.channel_layer.group_send(self.room_name, {"type": "user_disconnection"})

    async def send_message(self, data: Dict[str, str]) -> None:
        message: str = data.get("message", None)
        username: str = data.get("username", None)
        await self.preprocess_message(message, self.user)
        # await save_message(message, self.user.get("user_id", None))

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
            decrypted_message: str = decrypt_with_symmetric_key(message, self.symmetric_key) # <-- decrypted content of last message
            encrypted_messages: List[Dict[str, Any]] = await get_all_messages_from_user(self.user.get("user_id", None))
            print("printing encrypted messages from consumers", encrypted_messages)
            decrypted_messages: List[Dict[str, Any]] = [{"message": decrypted_message}] # <-- only the first one is not enriched (cuz its the last one)

            for encrypted_message in encrypted_messages:
                decrypted_content: str = decrypt_with_symmetric_key(encrypted_message.get("message", None), self.symmetric_key)
                encrypted_message["message"] = decrypted_content
                decrypted_messages.append(encrypted_message)
            
            print("printing decrypted messages from consumers", decrypted_messages)
            enriched_message: Dict[str, Any] = await get_enriched_message(decrypted_messages)

            answer: str = await generate_answer_from_enriched_message(enriched_message, decrypted_messages, self.user.get("user_id", None))
            
            if answer == "END":
                await set_user_status(self.user, "Disconnected")
                await self.close()
                return

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
        encrypted_message: str = event.get("message", None)
        encrypted_username: str = event.get("username", None)

        is_myself: bool = self.user.get("username", None) == decrypt_with_symmetric_key(encrypted_username, self.symmetric_key)

        if is_myself or "virtual_room" in self.room_name:
            await self.send(text_data=json.dumps({"username": encrypted_username, "message": encrypted_message}))
        else:
            decrypted_username: str = decrypt_with_symmetric_key(encrypted_username, self.new_symmetric_key)
            decrypted_message: str = decrypt_with_symmetric_key(encrypted_message, self.new_symmetric_key)
            re_encrypted_username: str = encrypt_with_symmetric_key(decrypted_username, self.symmetric_key)
            re_encrypted_message: str = encrypt_with_symmetric_key(decrypted_message, self.symmetric_key)
            await self.send(text_data=json.dumps({"username": re_encrypted_username, "message": re_encrypted_message}))

    async def assign_room(self) -> None:
        if self.user.get("is_worker", None) == "True":
            self.room_name: str = f"{self.user.get('username', None)}_waiting_room"
            await self.listen_to_rabbit()
            celery_task: AsyncResult = await sync_to_async(check_for_non_workers_task.delay)(self.user.get("username", None))
            self.celery_task_id: str = celery_task.id
        else:
            self.room_name: str = f"{self.user.get('username', None)}_virtual_room"

        await self.channel_layer.group_add(self.room_name, self.user.get("channel_name", None))

    async def process_non_worker_assignment(self, worker_username: str, free_non_worker: Dict[str, Any]):
        AsyncResult(self.celery_task_id).revoke(terminate=True)
        free_non_worker_username: str = decrypt_with_symmetric_key(free_non_worker.get("username", ""), self.symmetric_key)
        new_room_name: str = f"{worker_username}_{free_non_worker_username}_real_room"
        # cannot send worker's symmetric key encrypted with itself, cuz then the non worker would need to have the worker's symmetric key anyways
        await self.channel_layer.group_send(
            f"{free_non_worker_username}_virtual_room",
            {"type": "update_room_name", "new_room_name": new_room_name, "new_symmetric_key": self.symmetric_key},
        )
        # this must be done after update_room_name is called and therefore the worker receives the non worker symmetric key
        self.room_name: str = new_room_name
        await self.channel_layer.group_discard(
            f"{free_non_worker_username}_virtual_room",
            free_non_worker.get("channel_name", None),
        )
        await set_user_status(self.user, "Busy")
        await set_user_status(free_non_worker, "Busy")
        await self.channel_layer.group_add(new_room_name, self.user.get("channel_name", None))
        await self.channel_layer.group_add(new_room_name, free_non_worker.get("channel_name", None))

    async def update_room_name(self, event: Dict[str, Any]) -> None:
        self.room_name: str = event.get("new_room_name", None)
        self.new_symmetric_key: bytes = event.get("new_symmetric_key", None)
        # now the non worker has both symmetric keys, but the worker only has his, so he must receive the non worker's
        worker_username: str = event.get("new_room_name", "").split("_")[0]
        await self.channel_layer.group_send(
            f"{worker_username}_waiting_room",
            {"type": "update_symmetric_key", "new_symmetric_key": self.symmetric_key},
        )

    async def update_symmetric_key(self, event: Dict[str, Any]) -> None:
        # this is the worker receiving the non worker's symmetric key
        self.new_symmetric_key: bytes = event.get("new_symmetric_key", None)

    async def preprocess_message(self, message: str, user: Dict[str, Any]):
        await self.send(text_data=json.dumps({"error": "Message must not be empty."})) if not message else None
        # if theres an error with the token and i have to send the error response, i cannot do it before here (in get_user_from_token or in connect)
        # cuz the channel isnt even open
        await self.send(text_data=json.dumps(user)) if "error" in user else None
        # TODO: maybe add more checks

    async def user_disconnection(self, event: Dict[str, str]) -> None:
        if self.user.get("is_worker", None) == "True":
            # if the non worker disconnected, the worker goes back to pinging
            await set_user_status(self.user, "Free")
            await self.assign_room()
        else:
            # if the worker disconnected, the non worker closes the connection
            await set_user_status(self.user, "Disconnected")
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
