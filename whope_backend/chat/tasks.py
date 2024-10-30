import json
import random
from typing import Dict

from aio_pika import (
    Channel,
    DeliveryMode,
    Message,
    Queue,
    RobustConnection,
    connect_robust,
)
from asgiref.sync import async_to_sync
from celery import shared_task
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase

from whope_backend.settings import RABBITMQ_URI, get_motor_db


# bind=true allows to use "self"
@shared_task(bind=True, default_retry_delay=1, max_retries=None)
def check_for_non_workers_task(self, worker_username) -> None:
    free_non_worker: Dict[str, str] = async_to_sync(get_free_non_worker)()
    if free_non_worker:
        print("A")
        async_to_sync(send_message_to_rabbit)(worker_username, free_non_worker)
        return None
    else:
        raise self.retry()


async def get_free_non_worker() -> Dict[str, str]:
    db: AsyncIOMotorDatabase = await get_motor_db()
    users: AsyncIOMotorCollection = db["users"]
    non_workers: Dict[str, str] = await users.find(
        {"is_worker": "False", "status": "Free"}
    ).to_list(None)
    if non_workers:
        return random.choice(non_workers)
    return None


async def send_message_to_rabbit(
    worker_username: str, free_non_worker: Dict[str, str]
) -> None:
    print("C")
    connection: RobustConnection = await connect_robust(RABBITMQ_URI)
    async with connection:
        channel: Channel = await connection.channel()

        queue: Queue = await channel.declare_queue("assignment_queue", durable=True)
        # remove ObjectId from free_non_worker because its not encodable
        free_non_worker.pop("_id")
        free_non_worker.pop("messages")
        message: Dict[str, str] = {
            "worker_username": worker_username,
            "free_non_worker": free_non_worker,
        }
        print(f"mandando mensaje: {message}")
        await channel.default_exchange.publish(
            Message(
                body=json.dumps(message).encode(),
                delivery_mode=DeliveryMode.PERSISTENT,
            ),
            routing_key=queue.name,
        )
