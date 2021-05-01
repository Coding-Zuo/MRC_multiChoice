# -*- coding:utf-8 -*-
import csv
import copy
import json
import jsonlines


# 用于简单序列分类的单个训练/测试示例。
class InputExample(object):
    """
    guid：示例的唯一ID。
    text_a：字符串。第一个序列的未标记化文本。对于系列任务，则必须仅指定此序列。
    text_b：(可选)字符串。第二序列的未标记化文本。 只能为序列对任务指定。
    label：(可选)字符串。示例的标签。为train和dev指定，但不为test示例指定。
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputExample_MCinQA(object):
    """
    不含段落(仅限问题和候选)任务的多选题RC的单个train/dev/test示例
    """

    def __init__(self, guid, question, choices=[], label=None, q_id=None):
        self.guid = guid
        self.question = question
        self.choices = choices
        self.label = label
        self.q_id = q_id

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    def __init__(self, example_id, input_ids, attention_mask, token_type_ids, label_id):
        self.example_id = example_id
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id


class InputFeature_MCBase(object):
    """多项选择RC任务中示例的单一功能集"""

    def __init__(self, example_id, choice_features, lable_id):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choice_features
        ]
        self.label_id = lable_id


def select_field_MC(features, field):
    return [
        [
            choice[field]
            for choice in features.choices_features
        ]
        for feature in features
    ]


class DataProcessor(object):
    """序列分类数据集的数据转换器的基类。"""

    def get_example_from_tensor_dict(self, tensor_dict):
        """
        从具有张量的字典中获取示例
        参数：
        tensor_dict：键和值应该与对应的Glue匹配
        TensorFlow_DataSet示例。
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """
        获取train的InputExamples的集合
        """
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """
        获取dev集的InputExamples的集合
        """
        raise NotImplementedError()

    def get_labels(self):
        """获取此数据集的标签列表。"""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """读取制表符分隔值文件。"""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
        return lines

    @classmethod
    def _read_csv_with_delimiter(cls, input_file, delimiter=","):
        lines = []
        with open(input_file, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter)
            for line in reader:
                lines.append(line)
        return lines

    @classmethod
    def _read_csv(cls, input_file):
        """read csv"""
        lines = []
        with open(input_file, 'r', encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                lines.append(row)
        return lines

    @classmethod
    def _read_jsonl(cls, input_file):
        """read jsonl file"""
        lines = []
        with open(input_file, mode='r') as json_file:
            reader = jsonlines.Reader(json_file)
            for instance in reader:
                lines.append(instance)
        return lines
