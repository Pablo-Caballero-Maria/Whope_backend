from bson import ObjectId
import json
from typing import Any, Dict
from channels.generic.websocket import AsyncWebsocketConsumer
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
from whope_backend.settings import get_motor_db, RABBITMQ_URI
from datetime import datetime, timezone
import asyncio
from urllib.parse import parse_qs
from chat.tasks import check_for_non_workers_task
from celery.result import AsyncResult
from asgiref.sync import sync_to_async
import logging
import aio_pika

# room_name is the name of the room that the user is connected to.
# room_group_name groups all the users connected to the a room.
# channel_name is the unique identifier of the connection between the user and the room
# channel_name will be the username of the user if the user
# channel_layer.group allows to add and remove a channel from a group
class ChatConsumer(AsyncWebsocketConsumer):

    async def connect(self) -> None:
        query_string: str = self.scope['query_string'].decode('utf-8')
        query_params: Dict[str, Any] = parse_qs(query_string)
        self.token: str = query_params.get('token', [None])[0]
        self.user: Dict[str, str] = await self.get_user_from_token(self.token)
        # channel_name is automatically set and it needs to be saved so than the worker can add the non_worker to the room
        # once he finds one
        await self.save_channel_name(self.user, self.channel_name)
        # this is just initialization (it must be defined):
        self.room_name: str = None
        # users have 3 status: Free (non worker chatting with ai/worker waiting), Busy (worker chatting with non worker or viceversa),
        # Disconnected (not connected to any ws)
        await self.set_user_status(self.user, "Free")
        await self.assign_room()
        await self.accept()

    async def disconnect(self, close_code: int) -> None:
        AsyncResult(self.celery_task_id).revoke(terminate=True) if self.user.get('is_worker', False) == 'True' else None
        await self.rabbit_connection.close() if self.user.get('is_worker', False) else None
        await self.channel_layer.group_discard(self.room_name, self.user.get('channel_name', None))
        await self.save_channel_name(self.user, None)
        await self.set_user_status(self.user, "Disconnected")
        # this message will only be received by the OTHER user
        await self.channel_layer.group_send(self.room_name, {'type': 'user_disconnection'})

    async def receive(self, text_data: str) -> None:
        data: Dict[str, str] = json.loads(text_data)
        message: str = data.get('message', None)
        await self.preprocess_message(message, self.user) 
        await self.save_message(message)

        if "virtual_room" in self.room_name:
            # TODO: use ai bot here
            await self.send(text_data=json.dumps({'message': f'Your message {message} has been sent.'})) 
        else:
            await self.channel_layer.group_send(
                self.room_name,
                    {
                        'type': 'chat_message',
                        'username': self.user.get('username', None),
                        'message': message
                    }
                )

    async def chat_message(self, event: Dict[str, str]) -> None:
        await self.send(text_data=json.dumps({
            'username': event.get('username', None),
            'message': event.get('message', None)
        }))

    async def save_message(self, message_content: str) -> None:
        users: AsyncIOMotorCollection = await self.get_users()
        message: Dict[str, str] = {"message": message_content, "created_at": datetime.now(timezone.utc)}
        await users.update_one({"_id": ObjectId(self.user.get('user_id', None))}, {"$push": {"messages": message}})
        
    async def get_user_from_token(self, token: str) -> Dict[str, str]:
        from rest_framework_simplejwt.tokens import UntypedToken
        from rest_framework_simplejwt.exceptions import TokenError
        if not token:
            return {'error': 'Token must not be empty.'}
       
        try:
            decoded_token: UntypedToken = UntypedToken(token)
            user_id: str = decoded_token.get('user_id', None)
            username: str = decoded_token.get('username', None)
            is_worker: bool = decoded_token.get('is_worker', False)
            return {"user_id": user_id, "username": username, 'is_worker': is_worker}

        except TokenError:
            return {'error': 'Invalid token.'}

    async def assign_room(self) -> None:
        if self.user.get('is_worker', False) == 'True':
            self.room_name: str = f"{self.user.get('username', None)}_waiting_room"
            await self.listen_to_rabbit()
            # the celery celery_task is asynchronous while it runs, but the fact that we must wait for it to finish
            # makes it synchronous in practice, which blocks the asynchronous consumer, preventing it from listening
            # to other events. thus the necessity of monitoring it asynchronously by using asyncio.create_celery_task(monitor_celery_task)
            # which will stop once check_for_non_workers finishes (returning None)
            # we can monitor asynchronously the task itself because delay returns an objcet with a property "result" that can be "ready"
            # unlike rabbit, where that checking of props must be done by us
            # "delay" returns an AsyncResult object, which cannot be used with await
            celery_task: AsyncResult = await sync_to_async(check_for_non_workers_task.delay)(self.user.get('username', None))
            self.celery_task_id: str = celery_task.id
            # we hace to call self.monitor_celery_task(celery_task) with asyncio.create_task instead of await because
            # then, it wouldnt be cancelable
            asyncio.create_task(self.monitor_celery_task(celery_task))
            print("B")
        else:
            self.room_name: str = f"{self.user.get('username', None)}_virtual_room"

        await self.channel_layer.group_add(self.room_name, self.user.get('channel_name', None))

    async def process_non_worker_assignment(self, worker_username, free_non_worker):
        # if this code is reached, then the celery task is already killed
        await self.rabbit_connection.close()
        new_room_name: str = f"{worker_username}_{free_non_worker['username']}_real_room"
        self.room_name: str = new_room_name
        await self.channel_layer.group_send(f"{free_non_worker['username']}_virtual_room", {'type': 'update_room_name', 'new_room_name': new_room_name})
        await self.channel_layer.group_discard(f"{free_non_worker['username']}_virtual_room", free_non_worker.get('channel_name', None))
        await self.set_user_status(self.user, "Busy")
        await self.set_user_status(free_non_worker, "Busy")
        await self.channel_layer.group_add(new_room_name, self.user.get('channel_name', None))
        await self.channel_layer.group_add(new_room_name, free_non_worker.get('channel_name', None))

    async def set_user_status(self, user: Dict[str, str], status: str) -> None:
        users: AsyncIOMotorCollection = await self.get_users()
        await users.update_one(
            {"_id": ObjectId(user.get('user_id', None))},
            {"$set": {"status": status}}
        )

    async def save_channel_name(self, user: Dict[str, str], channel_name: str) -> None:
        users: AsyncIOMotorCollection = await self.get_users()
        await users.update_one(
            {"_id": ObjectId(user.get('user_id', None))},
            {"$set": {"channel_name": channel_name}}
        )
        # so that locally its also updated without having to call db again
        self.user['channel_name']: str = channel_name

    async def update_room_name(self, event: Dict[str, str]) -> None:
        self.room_name: str = event.get('new_room_name', None)

    async def get_users(self) -> AsyncIOMotorCollection:
        db: AsyncIOMotorDatabase = await get_motor_db()
        return db["users"]

    async def preprocess_message(self, message: str, user: Dict[str, str]):
        await self.send(text_data=json.dumps({'error': 'Message must not be empty.'})) if not message else None
        # if theres an error with the token and i have to send the error response, i cannot do it before here (in get_user_from_token or in connect)
        # cuz the channel isnt even open
        await self.send(text_data=json.dumps(user)) if 'error' in user else None
        # TODO: maybe add more checks

    async def user_disconnection(self, event: Dict[str, str]) -> None:
        if self.user.get('is_worker', False) == 'True':
            AsyncResult(self.celery_task_id).revoke(terminate=True)
            await self.rabbit_connection.close()
            # if the non worker disconnected, the worker goes back to pinging
            await self.set_user_status(self.user, "Free")
            await self.assign_room()
        else:
            # if the worker disconnected, the non worker closes the connection
            await self.set_user_status(self.user, "Disconnected")
            await self.close()

    async def listen_to_rabbit(self) -> None:
        # Configurar el logger para aio_pika
        pika_logger = logging.getLogger('aio_pika')
        pika_logger.setLevel(logging.ERROR)

        # Conectar asincrónicamente a RabbitMQ
        connection: aio_pika.RobustConnection = await aio_pika.connect_robust(RABBITMQ_URI)
        self.rabbit_connection: aio_pika.RobustConnection = connection
        channel: aio_pika.Channel = await self.rabbit_connection.channel()
        await channel.set_qos(prefetch_count=1)  # Controla la cantidad de mensajes sin procesar permitidos

        # Declarar la cola y configurar el consumidor
        queue: aio_pika.Queue = await channel.declare_queue("assignment_queue", durable=True)

        async def on_message(message: aio_pika.IncomingMessage) -> None:
            async with message.process():
                data: Dict[str, str] = json.loads(message.body)
                worker_username: str = data.get('worker_username', None)
                free_non_worker: str = data.get('free_non_worker', None)

                if worker_username == self.user.get('username'):
                    await self.process_non_worker_assignment(worker_username, free_non_worker)
                    await self.rabbit_connection.close()  # Cerrar la conexión una vez procesado el mensaje
        # Iniciar la escucha asincrónica
        await queue.consume(on_message)

    async def monitor_celery_task(self, celery_task_id: str) -> None:
        while True:
            # paradoxically, AsyncResult is not asynchronous, so we wrap it in sync_to_async to use it inside monitor_celery_task
            # which is asynchronous since its a (non blocking) asyncio celery_task
            celery_task_result: None = await sync_to_async(self.get_celery_task_result)(celery_task_id)
            if celery_task_result:
                break
            else:
                await asyncio.sleep(1)

    def get_celery_task_result(self, celery_task_id: str) -> None:
        result: AsyncResult = AsyncResult(celery_task_id)
        if result.ready():
            return result.result
        return None

