from pymongo.database import Collection, Database
from celery import shared_task
from whope_backend.settings import get_pymongo_db, RABBITMQ_URI
import random
from typing import Dict
import json
import aio_pika
from asgiref.sync import async_to_sync

# bind=true allows to use "self"
@shared_task(bind=True, default_retry_delay=1, max_retries=None)
def check_for_non_workers_task(self, worker_username) -> None:
    free_non_worker: Dict[str, str] = get_free_non_worker()  
    if free_non_worker:
        async_to_sync(send_message_to_rabbit)(worker_username, free_non_worker)
        return None
    else:
        print("reintentando")
        raise self.retry()

def get_free_non_worker() -> Dict[str, str]:
    db: Database = get_pymongo_db()
    users: Collection = db["users"]
    non_workers: Dict[str, str] = users.find({"is_worker": "False", "status": "Free"}).to_list(None)
    print(non_workers)
    if non_workers:
        return random.choice(non_workers)
    return None

async def send_message_to_rabbit(worker_username: str, free_non_worker: Dict[str, str]) -> None:
    # Conectar asincrónicamente a RabbitMQ
    connection: aio_pika.RobustConnection = await aio_pika.connect_robust(RABBITMQ_URI)
    async with connection:
        channel: aio_pika.Channel = await connection.channel()
        
        # Declarar la cola en la que se enviará el mensaje
        queue: aio_pika.Queue = await channel.declare_queue("assignment_queue", durable=True)
        
        # Preparar el mensaje en formato JSON
        message: Dict[str, str] = {
            'worker_username': worker_username,
            'free_non_worker': free_non_worker
        }
        
        # Publicar el mensaje en la cola con la propiedad de persistencia
        await channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(message).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT
            ),
            routing_key=queue.name
        )
