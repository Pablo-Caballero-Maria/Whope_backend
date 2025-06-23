from datetime import datetime, timezone
from typing import Any, Dict, List

from bson.objectid import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from utils.crypto_utils import (
    decrypt_with_private_key,
    decrypt_with_symmetric_key,
    encrypt_with_symmetric_key,
)

from whope.settings import PRIVATE_KEY_BYTES, get_motor_db


async def save_message(message_content: str, user_id: str) -> None:
    message: Dict[str, Any] = {
        "message": message_content,
        "created_at": datetime.now(timezone.utc),
    }
    users: AsyncIOMotorCollection = await get_users()
    await users.update_one(
        {"_id": ObjectId(user_id)},
        {"$push": {"messages": message}},
    )


async def save_message_AI(enriched_message: Dict[str, str], user_id) -> None:
    users: AsyncIOMotorCollection = await get_users()
    await users.update_one(
        {"_id": ObjectId(user_id)},
        {"$push": {"messages": enriched_message}},
    )


async def get_users() -> AsyncIOMotorCollection:
    db: AsyncIOMotorDatabase = await get_motor_db()
    users: AsyncIOMotorCollection = db["users"]
    return users


async def get_user_from_token(token: str) -> Dict[str, Any]:
    from rest_framework_simplejwt.tokens import UntypedToken

    decoded_token: UntypedToken = UntypedToken(token)
    user_id: str = decoded_token.get("user_id", None)
    username: str = decoded_token.get("username", None)
    is_worker: bool = decoded_token.get("is_worker", None)
    return {"user_id": user_id, "username": username, "is_worker": is_worker}


async def set_user_status(user: Dict[str, Any], status: str) -> None:
    users: AsyncIOMotorCollection = await get_users()
    await users.update_one({"_id": ObjectId(user.get("user_id", None))}, {"$set": {"status": status}})


async def save_channel_name(user: Dict[str, Any], channel_name: str) -> None:
    users: AsyncIOMotorCollection = await get_users()
    await users.update_one({"_id": ObjectId(user.get("user_id", None))}, {"$set": {"channel_name": channel_name}})


async def store_user(data: Dict[str, Any]) -> Dict[str, str]:
    username: str = data.get("username", None)
    password: str = data.get("password", None)
    is_worker: bool = data.get("is_worker", False)

    users: AsyncIOMotorCollection = await get_users()
    user: Dict[str, Any] = await users.find_one({"username": username})

    if user is not None:
        return {"error": "User already exists."}

    await users.insert_one(
        {
            "username": username,
            "password": password,
            "is_worker": is_worker,
            "status": "Disconnected",
        }
    )

    return {"success": "User created successfully."}


async def generate_token(data: Dict[str, Any]) -> Dict[str, str]:
    encrypted_username: str = data.get("encrypted_username", "")
    encrypted_password: str = data.get("encrypted_password", "")
    encrypted_symmetric_key: str = data.get("encrypted_symmetric_key", "")
    symmetric_key: bytes = decrypt_with_private_key(encrypted_symmetric_key.encode("utf-8"), PRIVATE_KEY_BYTES)

    users: AsyncIOMotorCollection = await get_users()
    user: Dict[str, Any] = await users.find_one({"username": encrypted_username})

    if encrypted_password != user.get("password", None):
        return {"error": "Invalid credentials."}

    from rest_framework_simplejwt.tokens import RefreshToken

    refresh: RefreshToken = RefreshToken()
    decrypted_username: str = decrypt_with_symmetric_key(encrypted_username, symmetric_key)
    # no need to store the username encrypted inside the token since the token itself will be encrypted
    refresh["username"] = decrypted_username
    refresh["is_worker"] = user.get("is_worker", None)
    refresh["user_id"] = str(user.get("_id", None))
    access_token: str = str(refresh.access_token)
    tokens: Dict[str, Any] = {
        "refresh": str(refresh),
        "access": encrypt_with_symmetric_key(access_token, symmetric_key),
    }
    return tokens


async def get_all_messages_from_user(user_id: str) -> List[Dict[str, str]]:
    users: AsyncIOMotorCollection = await get_users()
    user: Dict[str, Any] = await users.find_one({"_id": ObjectId(user_id)})
    messages: List[Dict[str, Any]] = user.get("messages", [])
    return messages
