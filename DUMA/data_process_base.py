# -*- coding:utf-8 -*-
import os
import six
import csv
import logging
import json
import torch
import copy
import jsonlines

logger = logging.getLogger("Chn")


# 用于简单序列分类的单个训练/测试示例。
class InputExample(object):
    def __init__(self, guid, text_a, text_b, label=None):
        self.guid = guid  # 示例的唯一ID。
        self.text_a = text_a  # 第一个序列的文本进行了非标记化。 单序列至少指定他
        self.text_b = text_b  # 第二序列的未标记化文本。只能为序列对任务指定
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


# 一组单一的数据特征。
class InputFeatures(object):
    def __init__(self, example_id, input_ids, attention_mask, token_type_ids, label_id):
        self.example_id = example_id
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id


# 多项选择RC任务中示例的单一特征集
class InputFeature_MultiChoice(object):
    def __init__(self, example_id, choice_features, label_id):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choice_features
        ]
        self.label_id = label_id


class DataProcessor(object):
    """序列分类数据集的数据转换器的基类"""

    def get_example_from_tensor_dict(self, tensor_dict):
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotecher=None):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotecher=quotecher)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_csv_with_delimiter(cls, input_file, delimiter=','):
        lines = []
        with open(input_file, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter)
            for line in reader:
                lines.append(line)
        return lines

    @classmethod
    def _read_csv(cls, input_file):
        "Read a csv file"
        lines = []
        with open(input_file, 'r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                lines.append(row)
        return lines

    # read ReCAM jsonl data
    @classmethod
    def _read_jsonl(cls, input_file):
        "Read a jsonl file"
        lines = []
        with open(input_file, mode='r') as json_file:
            reader = jsonlines.Reader(json_file)
            for instance in reader:
                lines.append(instance)
        return lines


def batch_pad(batch_data, args, pad=0):
    seq_len = [len(i) for i in batch_data]
    max_len = max(seq_len)
    if max_len > args.max_len:
        max_len = args.max_len
    out = []
    for line in batch_data:
        if len(line) < max_len:
            out.append(line + [pad] * (max_len - len(line)))
        else:
            out.append(line[:args.max_len])
    return torch.tensor(out, dtype=torch.long).cuda()
