
import os
from celery import Celery
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB = os.getenv("REDIS_DB", "0")

REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

celery_app = Celery(
    "docreader_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["worker.tasks"]
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    worker_prefetch_multiplier=1, # Important for GPU tasks
    worker_max_tasks_per_child=1, # Important for GPU memory management
)

# Optional: Configure GPU visibility for workers
# This can also be set as an environment variable before starting the worker
# e.g., CUDA_VISIBLE_DEVICES=0 celery -A worker.celery_app worker
# For multi-GPU setups, you might want to assign specific GPUs to specific workers
# or use a resource manager like Kubernetes with NVIDIA GPU Operator.

