# -*- coding:utf-8 -*-
import os
import logging
import six
from bertBaseDistribute.LongFormer.dataset.dataset_base import DataProcessor
from bertBaseDistribute.LongFormer.dataset.dataset_base import InputExample_MCinQA, InputFeature_MCBase

logger = logging.getLogger("haihua")


class Chn_Processor(DataProcessor):

    def get_train_examples(self, data_file, file_name=None):
        input_file = os.path.join(data_file, file_name)
        logger.info(f"Load haihua train data from:[{input_file}]")
        return self._create_examples(
            records=self._read_jsonl(input_file),
            set_type='train'
        )

    def get_dev_examples(self, data_dir, file_name=None):
        input_file = os.path.join(data_dir, file_name)
        logger.info(f"Load haihua Dev data from:[{input_file}]")
        return self._create_examples(
            records=self._read_jsonl(input_file),
            set_type='dev'
        )

    def get_test_examples(self, data_dir, file_name=None):
        input_file = os.path.join(data_dir, file_name)
        logger.info(f"Load haihua Dev data from [{input_file}]")
        return self._create_examples(
            records=self._read_jsonl(input_file),
            set_type='test'
        )

    def get_labels(self):
        return [0, 1, 2, 3, 4]

    def _create_examples(self, records, set_type='train'):
        """为训练集集和测试集创建实例"""
        examples = []
        for (i, line) in enumerate(records):
            record = line
            ex_id = str(i)
            guid = "%s-%s" % (set_type, ex_id)
            # article = record['Content']
            # question = record['Question']
            # choices = record['Choices'][2:-2].split('\', \'')
            # if len(choices) < 4:  # 如果选项不满四个，就补“不知道”
            #     for i in range(4 - len(choices)):
            #         choices.append('D．不知道')
            #
            # choices_opt = []
            # for choice in choices:
            #     choices_opt.append(replace_placeholder(question, choice))
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
                InputExample_MCinQA(guid=guid, question=article, choices=[opt1, opt2, opt3, opt4], label=label,
                                    q_id=q_id))
        return examples


def replace_placeholder(str, opt):
    list = str.split(' ')
    for i in range(len(list)):
        if list[i] == '（）':
            list[i] = opt
    final_opt = " ".join(list)
    return final_opt


def convert_examples_to_features(examples, label_list, tokenizer, max_length=512,
                                 pad_on_left=False, pad_token=0, pad_token_segment_id=0, mask_padding_with_zero=True,
                                 is_training=True):
    """将example转换为bert输入"""
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for example_index, example in enumerate(examples):
        context_tokens = example.question
        choices_features = []
        for ending_index, ending in enumerate(example.choices):
            context_tokens_choice = context_tokens[:]
            ending_tokens = ending

            inputs = tokenizer.encode_plus(
                text=ending_tokens,  # 要编码的可选第1序列
                text_pair=context_tokens_choice,  # 要编码的可选第二序列
                add_special_tokens=True,  # for [CLS] and [SEP] 是否使用相对于其模型的特殊令牌对序列进行编码。
                max_length=max_length,
                return_overflowing_tokens=True  # 是否返回溢出令牌序列。
            )

            input_ids, token_type_ids = inputs['input_ids'], inputs['token_type_ids']
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            padding_length = max_length - len(input_ids)

            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length

            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'])
            choices_features.append((tokens, input_ids, attention_mask, token_type_ids))

        label_id = label_map[example.label] if example.labal is not None else -1

        if example_index < 1:
            logger.info("*** Example ***")
            logger.info(f"example_id: {example.q_id}")
            for choice_idx, (tokens, input_ids, attention_mask, token_type_ids) in enumerate(choices_features):
                logger.info(f"choice: {choice_idx}")
                logger.info(f"tokens: {' '.join(tokens)}")
                logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
                logger.info(f"input_mask: {' '.join(map(str, attention_mask))}")
                logger.info(f"segment_ids: {' '.join(map(str, token_type_ids))}")
                logger.info(f"label: {label_id}")

        if example_index % 2000 == 0:
            logger.info(f"convert: {example_index}")

        features.append(
            InputFeature_MCBase(example_id=example.q_id, choices_features=choices_features, label_id=label_id))
    return features


if __name__ == '__main__':
    import bertBaseDistribute.args as args

    args = args.init_arg_parser()
    processor = Chn_Processor()
    # processor.get_train_examples(args.data_dir, 'train.csv')
    processor.get_train_examples(args.data_dir, 'train_baseline.json')
