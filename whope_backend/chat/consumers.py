import json
from typing import Any, Dict
from channels.generic.websocket import AsyncWebsocketConsumer
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
from whope_backend.settings import get_motor_db
from datetime import datetime, timezone

# room_name is the name of the room that the user is connected to.
# room_group_name groups all the users connected to the a room.
# channel_name is the unique identifier of the connection between the user and the room
# channel_layer.group allows to add and remove a channel from a group
class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self) -> None:
        self.room_name: str = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name: str = f"chat_{self.room_name}"
        
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code: int) -> None:
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data: str) -> None:
        data: Dict[str, Any] = json.loads(text_data)
        token: str = data.get('token', '')
        message: str = data.get('message', '')
        
        if not message:
            await self.send(text_data=json.dumps({'error': 'Message must not be empty.'}))
            return

        if not token:
            await self.send(text_data=json.dumps({'error': 'Token must not be empty.'}))
            return

        self.user: Dict[str, Any] = await self.get_user_from_token(token)
        if not self.user:
            await self.send(text_data=json.dumps({'error': 'Invalid token.'}))
            return

        await self.save_message(self.user['username'], message)

        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'username': self.user['username'],
                'message': message
            }
        )

    async def chat_message(self, event: Dict[str, Any]) -> None:
        await self.send(text_data=json.dumps({
            'username': event['username'],
            'message': event['message']
        }))

    async def save_message(self, username: str, message: str) -> None:
        messages: AsyncIOMotorCollection = await self.get_messages()

        await messages.insert_one({
            'room_name': self.room_name,
            'username': username,
            'message': message,
            'created_at': datetime.now(timezone.utc),
        })

    async def get_user_from_token(self, token: str) -> Dict[str, Any]:
        from rest_framework_simplejwt.tokens import UntypedToken
        
        decoded_token: UntypedToken = UntypedToken(token)
        user_id: str = decoded_token['user_id']
        username: str = decoded_token['username']

        return {'user_id': user_id, 'username': username}

    async def get_messages(self) -> AsyncIOMotorCollection:
        db: AsyncIOMotorDatabase = await get_motor_db()
        return db["messages"]
