import pytest
from channels.testing import WebsocketCommunicator
from django.test import TransactionTestCase
from whope_backend.asgi import application
from django.conf import settings
from typing import Any, Dict
import os

@pytest.fixture
def mongo_db():
    os.environ['TESTING'] = '1'
    db = settings.MONGO_DB
    db.drop_collection('users')
    yield db

@pytest.mark.asyncio
@pytest.mark.usefixtures("mongo_db")
class TestAuthenticationConsumer(TransactionTestCase): 

    async def test_dummy(self) -> None:
        assert 1 + 1 == 2

    async def websocket_connect(self) -> WebsocketCommunicator:
        communicator: WebsocketCommunicator = WebsocketCommunicator(application, "/ws/authentication/")
        connected, _ = await communicator.connect()
        assert connected
        return communicator

    async def test_register_user_success(self) -> None:
        communicator: WebsocketCommunicator = await self.websocket_connect()

        register_data: Dict[str, str] = {
            'action': 'register',
            'username': 'username_test',
            'password': 'password_test',
        }
        await communicator.send_json_to(register_data)
        
        response: str = await communicator.receive_json_from()
        assert response['message'] == 'User registered successfully.'

    async def test_register_user_already_exists(self) -> None:
        communicator: WebsocketCommunicator = await self.websocket_connect()

        register_data: Dict[str, str] = {
            'action': 'register',
            'username': 'username_test',
            'password': 'password_test',
        }
        await communicator.send_json_to(register_data)
        await communicator.receive_json_from()

        await communicator.send_json_to(register_data)
        response: str = await communicator.receive_json_from()
        assert response['error'] == 'User already exists.'

    async def test_login_user_success(self) -> None:
        communicator: WebsocketCommunicator = await self.websocket_connect()

        register_data: Dict[str, str] = {
            'action': 'register',
            'username': 'username_test',
            'password': 'password_test',
        }
        await communicator.send_json_to(register_data)
        await communicator.receive_json_from()  

        login_data: Dict[str, str] = {
            'action': 'login',
            'username': 'username_test',
            'password': 'password_test',
        }
        await communicator.send_json_to(login_data)

        response: str = await communicator.receive_json_from()
        assert response['message'] == 'Login successful.'
        assert 'tokens' in response

    async def test_login_user_invalid_credentials(self) -> None:
        communicator: WebsocketCommunicator = await self.websocket_connect()
        
        register_data: Dict[str, str] = {
            'action': 'register',
            'username': 'username_test',
            'password': 'password_test',
        }

        await communicator.send_json_to(register_data)
        await communicator.receive_json_from()

        login_data: Dict[str, str] = {
            'action': 'login',
            'username': 'username_test',
            'password': 'password_test_wrong',
        }
        await communicator.send_json_to(login_data)

        response: str = await communicator.receive_json_from()
        assert response['error'] == 'Invalid password.'

    async def test_login_user_missing_fields(self) -> None:
        communicator: WebsocketCommunicator = await self.websocket_connect()

        login_data: Dict[str, str] = {
            'action': 'login',
        }
        await communicator.send_json_to(login_data)

        response: str = await communicator.receive_json_from()
        assert response['error'] == 'Username and password are required.'

    async def test_register_user_missing_fields(self) -> None:
        communicator: WebsocketCommunicator = await self.websocket_connect()

        register_data: Dict[str, str] = {
            'action': 'register',
        }
        await communicator.send_json_to(register_data)

        response: str = await communicator.receive_json_from()
        assert response['error'] == 'Username and password are required.'
