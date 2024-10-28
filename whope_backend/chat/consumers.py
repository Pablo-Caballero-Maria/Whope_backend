from bson import ObjectId
import json
from typing import Any, Dict
from channels.generic.websocket import AsyncWebsocketConsumer
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
from whope_backend.settings import get_motor_db
from datetime import datetime, timezone
import random
import asyncio
from urllib.parse import parse_qs

# room_name is the name of the room that the user is connected to.
# room_group_name groups all the users connected to the a room.
# channel_name is the unique identifier of the connection between the user and the room
# channel_name will be the username of the user if the user
# channel_layer.group allows to add and remove a channel from a group
class ChatConsumer(AsyncWebsocketConsumer):

    async def connect(self) -> None:
        self.task = None
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
        # the cancel has to be here as well as in check_for_non_workers in case the worker has not found a non worker
        # (and thus canceled the task) before disconnecting
        self.task.cancel() if self.task else None
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
            task: asyncio.Task = asyncio.create_task(self.check_for_non_workers())
            self.task = task
        else:
            self.room_name: str = f"{self.user.get('username', None)}_virtual_room"
                
        await self.channel_layer.group_add(self.room_name, self.user.get('channel_name', None))

        
    async def check_for_non_workers(self) -> None:
        # only workers ping looking for non workers
        interval: float = 1
        while True:
            free_non_worker: Dict[str, str] = await self.get_free_non_worker()
            if free_non_worker:
                worker_username = self.user.get('username', None)
                non_worker_username = free_non_worker.get('username', None)
                await self.channel_layer.group_discard(self.room_name, self.user.get('channel_name', None))
                new_room_name: str = f"{worker_username}_{non_worker_username}_real_room"
                self.room_name: str = new_room_name # this has only been called by the worker, so we need to change this variable for the non worker
                await self.channel_layer.group_send(f"{non_worker_username}_virtual_room", {'type': 'update_room_name', 'new_room_name': new_room_name})
                await self.channel_layer.group_discard(f"{non_worker_username}_virtual_room", free_non_worker.get('channel_name', None))
                await self.set_user_status(self.user, "Busy")
                await self.set_user_status(free_non_worker, "Busy")
                await self.channel_layer.group_add(new_room_name, self.user.get('channel_name', None))
                await self.channel_layer.group_add(new_room_name, free_non_worker.get('channel_name', None))
                # cannot kill task before cuz the function would return too early without completing all steps
                self.task.cancel()
                break
            else:
                await asyncio.sleep(interval)

    async def set_user_status(self, user: Dict[str, str], status: str) -> None:
        users: AsyncIOMotorCollection = await self.get_users()
        await users.update_one(
            {"_id": ObjectId(user.get('user_id', None))},
            {"$set": {"status": status}}
        )

    async def get_free_non_worker(self) -> Dict[str, str]:
        # i have to make a new call to db to get the most updated version
        db: AsyncIOMotorDatabase = await get_motor_db()
        users: AsyncIOMotorCollection = db["users"]
        non_workers: Dict[str, str] = await users.find({"is_worker": "False", "status": "Free"}).to_list(None)
        if non_workers:
            return random.choice(non_workers)
        return None

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
            # if the non worker disconnected, the worker goes back to pinging
            await self.set_user_status(self.user, "Free")
            await self.assign_room()
        else:
            # if the worker disconnected, the non worker closes the connection
            await self.close()
