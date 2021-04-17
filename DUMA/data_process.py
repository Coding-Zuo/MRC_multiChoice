# -*- coding:utf-8 -*-
import os
import six
import logging
from .dataset_base import DataProcessor
from .dataset_base import InputExample_MCinQA, InputFeatures_MCBase

logger = logging.getLogger("Chn")


class Chn_Processor(DataProcessor):
    "Processor for the Chn dataset."

    def get_train_examples(self, data_dir, file_name=None):
        input_file = os.path.join(data_dir, "train.json")
        logger.info(f"Load Chn train data from: [{input_file}]")
        return self._create_examples(
            records=self._read_jsonl(input_file),
            set_type='train'
        )

    def get_dev_examples(self, data_dir, file_name=None):
        input_file = os.path.join(data_dir, "dev.json")
        logger.info(f"Load Chn Dev data from: [{input_file}]")
        return self._create_examples(
            records=self._read_jsonl(input_file),
            set_type='dev'
        )

    def get_test_examples(self, data_dir, file_name=None):
        input_file = os.path.join(data_dir, "test.json")  # TODO: test file path
        logger.info(f"Load Chn test data from: [{input_file}]")
        return self._create_examples(
            records=self._read_jsonl(input_file),
            set_type='test'
        )

    def get_labels(self):
        return [0, 1, 2, 3, 4]

    def _create_examples(self, records, set_type='train'):
        """Creates examples for the training and trial sets."""
        examples = []

        for (i, line) in enumerate(records):
            record = line
            ex_id = str(i)
            guid = "%s-%s" % (set_type, ex_id)
            article = record['article']
            question = record['question']

            opt1 = record['option_0']
            opt1 = replace_placeholder(question, opt1)
            opt2 = record['option_1']
            opt2 = replace_placeholder(question, opt2)
            opt3 = record['option_2']
            opt3 = replace_placeholder(question, opt3)
            opt4 = record['option_3']
            opt4 = replace_placeholder(question, opt4)
            if set_type == 'test':
                label = None
                q_id = record['q_id']
            else:
                label = record['label']
                q_id = 0

            examples.append(
                InputExample_MCinQA(
                    guid=guid,
                    question=article,
                    choices=[opt1, opt2, opt3, opt4],
                    label=label,
                    q_id=q_id
                )
            )
        return examples


# replace placeholder of question with options
def replace_placeholder(str, opt):
    list = str.split(' ')
    for i in range(len(list)):
        # if list[i] == '@placeholder':
        if list[i] == '（）':
            list[i] = opt
    final_opt = ' '.join(list)
    return final_opt
