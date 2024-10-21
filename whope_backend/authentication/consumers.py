import json
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from passlib.hash import bcrypt
from pymongo.collection import Collection
from typing import Dict, Any

class AuthenticationConsumer(AsyncWebsocketConsumer):
    async def connect(self) -> None:
        await self.accept()

    async def disconnect(self, close_code: int) -> None:
        pass

    async def receive(self, text_data: str) -> None:
        data: Dict[str, Any] = json.loads(text_data)
        action: str = data.get('action', '')

        if action == 'register':
            await self.register_user(data)
        elif action == 'login':
            await self.login_user(data)

    async def register_user(self, data: Dict[str, Any]) -> None:
        username: str = data.get('username', '')
        password: str = data.get('password', '')
        
        if not password or not username:
            await self.send(json.dumps({'error': 'Username and password are required.'}))
            return

        db: Collection = settings.MONGO_DB['users']

        user: Dict[str, Any] = db.find_one({'username': username})  

        if user is not None:
            await self.send(json.dumps({'error': 'User already exists.'}))
            return

        hashed_password: str = bcrypt.hash(password)
        db.insert_one({
            'username': username,
            'password': hashed_password
        })

        await self.send(json.dumps({'message': 'User registered successfully.'}))

    async def login_user(self, data: Dict[str, Any]) -> None:
        username: str = data.get('username', '')
        password: str = data.get('password', '')

        if not password or not username:
            await self.send(json.dumps({'error': 'Username and password are required.'}))
            return

        user: Dict[str, Any] = settings.MONGO_DB['users'].find_one({'username': username})
        if not bcrypt.verify(password, user['password']):
            await self.send(json.dumps({'error': 'Invalid password.'}))
            return
        
        from rest_framework_simplejwt.tokens import RefreshToken
        user_id: str = str(user['_id'])
        refresh: RefreshToken = RefreshToken()
        refresh['user_id'] = user_id

        tokens: Dict[str, str] = {
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        }

        await self.send(json.dumps({
            'message': 'Login successful.',
            'tokens': tokens
        }))
