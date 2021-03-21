import logging
import os

from celery import Celery

CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND')

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s|%(filename)s|%(lineno)s| %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger(__name__)
app = Celery(broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
