from pymongo.database import Collection, Database
from celery import shared_task
from whope_backend.settings import get_pymongo_db
import random
from typing import Dict
import json
import pika
from pika import BlockingConnection, ConnectionParameters 

# bind=true allows to use "self"
@shared_task(bind=True, default_retry_delay=1, max_retries=None)
def check_for_non_workers_task(self, worker_username) -> None:
    free_non_worker: Dict[str, str] = get_free_non_worker()  
    if free_non_worker:
        send_message_to_rabbitmq(worker_username, free_non_worker)  
        return None
    else:
        print("reintenando")
        raise self.retry()

def get_free_non_worker() -> Dict[str, str]:
    db: Database = get_pymongo_db()
    users: Collection = db["users"]
    non_workers: Dict[str, str] = users.find({"is_worker": "False", "status": "Free"}).to_list(None)
    print(non_workers)
    if non_workers:
        return random.choice(non_workers)
    return None

def send_message_to_rabbitmq(worker_username: str, free_non_worker: Dict[str, str]) -> None:
    connection: BlockingConnection = BlockingConnection(ConnectionParameters('localhost'))
    channel: pika.channel.Channel = connection.channel()
    channel.queue_declare(queue='assignment_queue', durable=True)
    
    message: Dict[str, str] = {
        'worker_username': worker_username,
        'free_non_worker': free_non_worker
    }
    # when the free non worker is found, a message is sent to rabbitmq (through pika) to communicate the result 
    channel.basic_publish(
        exchange='',
        routing_key='assignment_queue',
        body=json.dumps(message),
        properties=pika.BasicProperties(
            delivery_mode=2,  # makes message persistant
        )
    )
    connection.close()
