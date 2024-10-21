import json
from channels.generic.websocket import AsyncWebsocketConsumer
from whope_backend.settings import get_motor_db
from passlib.hash import bcrypt
from typing import Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection

class AuthenticationConsumer(AsyncWebsocketConsumer):
    async def connect(self) -> None:
        await self.accept()

    async def disconnect(self, close_code: int) -> None:
        pass

    async def receive(self, text_data: Dict[str, str]) -> None:
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
        
        db: AsyncIOMotorDatabase = await get_motor_db()
        users: AsyncIOMotorCollection = db["users"]
        user: Dict[str, Any] = await users.find_one({'username': username})  
         
        if user is not None:
            await self.send(json.dumps({'error': 'User already exists.'}))
            return
        
        hashed_password: str = bcrypt.hash(password)
        await users.insert_one({
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

        db: AsyncIOMotorDatabase = await get_motor_db()
        users: AsyncIOMotorCollection = db["users"]
        user: Dict[str, Any] = await users.find_one({'username': username})
        if not bcrypt.verify(password, user['password']):
            await self.send(json.dumps({'error': 'Invalid password.'}))
            return
        
        from rest_framework_simplejwt.tokens import RefreshToken
        user_id: str = str(user['_id'])
        refresh: RefreshToken = RefreshToken()
        refresh['user_id'] = user_id
        refresh['username'] = username

        tokens: Dict[str, str] = {
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        }

        await self.send(json.dumps({
            'message': 'Login successful.',
            'tokens': tokens
        }))
