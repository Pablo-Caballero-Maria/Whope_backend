from pymongo.database import Collection, Database
from celery import shared_task
from whope_backend.settings import get_pymongo_db
import random
from typing import Dict

# bind=true allows to use "self"
@shared_task(bind=True, default_retry_delay=1, max_retries=None)
def check_for_non_workers_task(self) -> Dict[str, str]:
    free_non_worker: Dict[str, str] = get_free_non_worker()  
    if free_non_worker:
       return free_non_worker
    else:
        raise self.retry()

def get_free_non_worker() -> Dict[str, str]:
    db: Database = get_pymongo_db()
    users: Collection = db["users"]
    non_workers: Dict[str, str] = users.find({"is_worker": "False", "status": "Free"}).to_list(None)
    print(non_workers)
    if non_workers:
        return random.choice(non_workers)
    return None
