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

from whope.settings import RABBITMQ_URI, USERS


# bind=true allows to use "self"
@shared_task(bind=True, default_retry_delay=1, max_retries=None)
def check_for_non_workers_task(self, worker_username: str) -> None:
    free_non_worker: Dict[str, str] = async_to_sync(get_free_non_worker)()
    if free_non_worker:
        async_to_sync(send_message_to_rabbit)(worker_username, free_non_worker)
        return None
    else:
        print("looking for non workers")
        raise self.retry()


async def get_free_non_worker() -> Dict[str, str]:
    non_workers: Dict[str, str] = await USERS.find({"is_worker": "False", "status": "Free"}).to_list(None)
    if non_workers:
        return random.choice(non_workers)
    return None


async def send_message_to_rabbit(worker_username: str, free_non_worker: Dict[str, str]) -> None:
    connection: RobustConnection = await connect_robust(RABBITMQ_URI)
    async with connection:
        channel: Channel = await connection.channel()

        queue: Queue = await channel.declare_queue(f"{worker_username}_assignment_queue")
        # remove ObjectId from free_non_worker because its not encodable
        free_non_worker.pop("_id")
        free_non_worker.pop("messages")
        message: Dict[str, str] = {
            "worker_username": worker_username,
            "free_non_worker": free_non_worker,
        }
        await channel.default_exchange.publish(
            Message(
                body=json.dumps(message).encode(),
            ),
            routing_key=queue.name,
        )
