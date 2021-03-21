import dataclasses
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Union, Iterable, List, Optional

sys.path.append('/home/src/')

from pytorch_transformers import BertTokenizer, BertForSequenceClassification
import spacy
import torch

from aiAnalyzer.actions.zero_shot.action_list import ActionList
from aiAnalyzer.analyzer import AiAnalyzer
from aitoolkit import Filters
from aitoolkit import RegexParse
from transformers import pipeline


@dataclasses.dataclass
class Meta:
    service: str = 'actions'
    ver: str = '1803211600'
    tag: str = '3.3.8'


@dataclasses.dataclass
class Result:
    result: list
    meta: 'Meta' = Meta()


class TrainingConfig:
    ACTIONS_PATH = Path('aiAnalyzer') / 'actions'
    BERT_PATH = ACTIONS_PATH / 'bert' / 'bert_model'
    SPACY_MODEL_NAME = 'en_core_web_sm'

    def __init__(self,
                 action_list: 'ActionList' = ActionList(),
                 classifier: 'pipeline' = pipeline(
                     'zero-shot-classification',
                     model=str(ACTIONS_PATH / 'zero_shot' / 'model')
                 ),
                 command_parser: 'RegexParse' = RegexParse(),
                 model: 'BertForSequenceClassification' = BertForSequenceClassification.from_pretrained(str(BERT_PATH)),
                 parser: 'Filters' = Filters(),
                 spacy_nlp: 'spacy.language.Language' = spacy.load(SPACY_MODEL_NAME),
                 tokenizer: 'BertTokenizer' = BertTokenizer.from_pretrained(str(BERT_PATH))):
        self.action_list = action_list
        self.classifier = classifier
        self.command_parser = command_parser
        model.to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model = model
        self.parser = parser
        self.spacy_nlp = spacy_nlp
        self.tokenizer = tokenizer


class Analyzer:
    SENTENCES_PER_ONE_ACTION = 50

    def __init__(self, payload: Dict, config: 'TrainingConfig'):
        self.payload = payload
        self.config = config
        self.kodama = payload.get('kodama')
        self._max_score_for_block: Union[int, float] = -1

    def analyze(self) -> Dict:
        response: Dict = {'response': [], 'exception': None}
        try:
            result = self._analyze()
            response['response'] = result
        except Exception as ex:
            response['exception'] = ex
            exc_type, _, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            raise ex
        finally:
            logging.info(response)

        return dataclasses.asdict(Result(result=result))

    def _analyze(self) -> List[Dict]:
        self._make_logs()
        texts = self.payload.get('texts', [])
        action_sentences = self._predict_action_sentences(texts)
        final_response = []

        for i, part in enumerate(texts):
            resp = self._predict_response(part)
            if resp is None:
                continue

            self._handle_actions(resp, i)
            final_response.append(self._append_action(resp, action_sentences))

        del self.config.model
        del self.config.tokenizer

        return final_response

    def _predict_action_sentences(self, texts: Iterable) -> List:
        action_sentences = []
        for part in texts:
            text, text_id = part.get('text', ''), part.get('text_id', '')
            action_sentence = self.config.action_list.predict(text, text_id)
            if action_sentence >= 0:
                action_sentences.append(action_sentence)
        return action_sentences

    def _predict_response(self, part) -> Optional[Dict]:
        if 'is_smalltalk' in part or 'is_question' in part:
            return None

        text, text_id = part.get('text', ''), part.get('text_id', '')
        if self.config.command_parser.is_command(text):
            return None

        analyzer = AiAnalyzer(
            text_id=text_id,
            model=self.config.model,
            tokenizer=self.config.tokenizer,
            spacy_nlp=self.config.spacy_nlp,
            parser=self.config.parser)
        resp = analyzer.predict(text)
        del analyzer
        return resp

    def _handle_actions(self, resp: Dict, i: int) -> None:
        resp_actions = resp.get('actions')
        if resp_actions:
            kodama_score = (self._get_kodama(resp['id'], 'time_group_rules') +
                            self._get_kodama(resp['id'], 'soft_commit_group_rules')) / 2
            for action in resp_actions:
                if not action['is_next_step']:
                    continue

                if kodama_score > self._max_score_for_block:
                    self._max_score_for_block = kodama_score
                    count_sentence = i + 1
                    if count_sentence % self.SENTENCES_PER_ONE_ACTION == 0:
                        resp['actions'] = [action]
                        self._max_score_for_block = -1
                    break
            else:
                resp['actions'] = []

    def _append_action(self, resp: Dict, action_sentences: List) -> Dict:
        if resp['id'] in action_sentences and self._get_kodama(resp['id'], 'follow_up_group_rules') == 1:
            resp = self.config.action_list.append_action(resp, self.config.classifier)
        return resp

    def _get_kodama(self, text_id, rule_type: str) -> Union[int, float]:
        if self.kodama is None or rule_type not in self.kodama:
            return 0

        actions_groups = self.kodama.get(rule_type)
        if not actions_groups:
            return 0

        actions_group = actions_groups[0]
        for part in actions_group['texts']:
            if part['text_id'] == text_id:
                return part['score']

        return 0

    @staticmethod
    def _make_logs() -> None:
        logging.info(f'Using GPU: {torch.cuda.is_available()}')
        logging.info(f'__Python VERSION:{sys.version}')
        logging.info(f'__pyTorch VERSION:{torch.__version__}')
        logging.info(f'__Number CUDA Devices:{torch.cuda.device_count()}')
        logging.info('__Devices')
        logging.info(f'Active CUDA Device: GPU{torch.cuda.current_device()}')
        logging.info(f'Available devices {torch.cuda.device_count()}')
        logging.info(f'Current cuda device {torch.cuda.current_device()}')
