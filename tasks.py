import os
from typing import Dict

from analyzer import Analyzer, TrainingConfig
from celery_app import app

TASK_NAME = os.getenv('TASK_NAME')


@app.task(name=TASK_NAME)
def analyze(data: Dict) -> Dict:
    return Analyzer(data.get('payload', {}), TrainingConfig()).analyze()
