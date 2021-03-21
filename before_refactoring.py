import os
import copy
import logging
from typing import Dict
from multiprocessing import Process, Queue
import spacy
from aitoolkit import Filters
from aitoolkit import RegexParse
from transformers import pipeline

from celery import Celery

CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND')
TASK_NAME = os.getenv('TASK_NAME')

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s|%(filename)s|%(lineno)s| %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
app = Celery(broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)


def get_kodama_time_score(payload, text_id):
    if not "kodama" in payload:
        return 0

    kodama = payload["kodama"]
    if not "time_group_rules" in kodama:
        return 0

    actions_groups = kodama["time_group_rules"]
    if not len(actions_groups) > 0:
        return 0

    actions_group = actions_groups[0]
    for part in actions_group["texts"]:
        if part["text_id"] == text_id:
            return part["score"]

    return 0


def get_kodama_soft_commit_score(payload, text_id):
    if not "kodama" in payload:
        return 0

    kodama = payload["kodama"]
    if not "soft_commit_group_rules" in kodama:
        return 0

    actions_groups = kodama["soft_commit_group_rules"]
    if not len(actions_groups) > 0:
        return 0

    actions_group = actions_groups[0]
    for part in actions_group["texts"]:
        if part["text_id"] == text_id:
            return part["score"]

    return 0


def get_kodama_follow_up_score(payload, text_id):
    if not "kodama" in payload:
        return 0

    kodama = payload["kodama"]
    if not "follow_up_group_rules" in kodama:
        return 0

    actions_groups = kodama["follow_up_group_rules"]
    if not len(actions_groups) > 0:
        return 0

    actions_group = actions_groups[0]
    for part in actions_group["texts"]:
        if part["text_id"] == text_id:
            return part["score"]

    return 0


def get_result(queue, data):
    response_of_child = {'response': [], 'exception': None}
    try:
        import sys
        sys.path.append('/home/src/')

        from aiAnalyzer.analyzer import AiAnalyzer
        from aiAnalyzer.actions.zero_shot.action_list import ActionList
        import torch
        from pytorch_transformers import BertTokenizer, BertForSequenceClassification

        logging.info(f'Using GPU: {torch.cuda.is_available()}')
        logging.info(f'__Python VERSION:{sys.version}')
        logging.info(f'__pyTorch VERSION:{torch.__version__}')
        logging.info(f'__Number CUDA Devices:{torch.cuda.device_count()}')
        logging.info(f'__Devices')
        logging.info(f'Active CUDA Device: GPU{torch.cuda.current_device()}')
        logging.info(f'Available devices {torch.cuda.device_count()}')
        logging.info(f'Current cuda device {torch.cuda.current_device()}')

        if torch.cuda.is_available():
            device = torch.device("cuda")  # change to gpu
        else:
            device = torch.device("cpu")

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained('aiAnalyzer/actions/bert/bert_model')
        tokenizer = BertTokenizer.from_pretrained('aiAnalyzer/actions/bert/bert_model')

        # Copy the model to the GPU.
        model.to(device)

        spacy_nlp = spacy.load("en_core_web_sm")
        parser = Filters()
        command_parser = RegexParse()
        action_list = ActionList()

        response = []
        action_sentences = []
        payload = data.get("payload", {})
        for part in payload["texts"]:
            if part.get("is_smalltalk", ""):
                continue
            if part.get("is_question", ""):
                continue
            text = part.get("text", "")
            text_id = part.get("text_id", "")

            if command_parser.is_command(text):
                continue

            analyzer = AiAnalyzer(text_id=text_id, model=model, tokenizer=tokenizer, spacy_nlp=spacy_nlp, parser=parser)
            result = analyzer.predict(text)
            action_sentence = action_list.predict(text, text_id)

            del analyzer
            response.append(result)

            if action_sentence >= 0:
                action_sentences.append(action_sentence)

        del model
        del tokenizer
        sentences_per_one_action = 50
        new_response = []
        max_score_for_block = -1
        picked_response = {}
        picked_position = 0
        count_sentence = 0
        position = 0
        for resp in response:
            count_sentence = count_sentence + 1

            if len(resp['actions']) > 0:
                kodama_score = (get_kodama_time_score(payload, resp['id']) +
                                get_kodama_soft_commit_score(payload, resp['id'])) / 2
                for action in resp['actions']:
                    if not action['is_next_step']:
                        continue

                    if kodama_score > max_score_for_block:
                        max_score_for_block = kodama_score
                        picked_response = copy.copy(resp)
                        picked_response['actions'] = [action]
                        picked_position = position

            new_resp = copy.copy(resp)
            new_resp['actions'] = []
            new_response.append(new_resp)

            if count_sentence % sentences_per_one_action == 0 and picked_response != {}:
                new_response[picked_position] = picked_response
                max_score_for_block = -1
                picked_response = {}
                picked_position = 0

            position = position + 1

        final_response = []
        classifier = pipeline("zero-shot-classification",
                              model="aiAnalyzer/actions/zero_shot/model")

        for resp in new_response:
            new_resp = copy.copy(resp)
            if resp['id'] in action_sentences and get_kodama_follow_up_score(payload, new_resp['id']) == 1:
                new_resp = action_list.append_action(new_resp, classifier)

            final_response.append(new_resp)

        response_of_child['response'] = final_response
    except Exception as ex:
        import sys, os
        response_of_child['exception'] = ex
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    finally:
        queue.put(response_of_child)


@app.task(name=TASK_NAME)
def analyze(data: Dict) -> Dict:
    qout = Queue()

    worker = Process(target=get_result, args=(qout, data))
    worker.start()
    result = qout.get()
    worker.join()
    logging.info(result)

    response = []
    if result['exception'] is not None:
        raise result['exception']
    else:
        response = result['response']

    return {
        "meta": {
            "service": "actions",
            "ver": "1803211600",
            "tag": "3.3.8"
        },
        "result": response
    }
