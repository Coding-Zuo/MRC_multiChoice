# -*- coding:utf-8 -*-
import os
import torch
import six
import logging
from bertBaseDistribute.args import init_arg_parser
from torch.utils.data import Dataset
from bertBaseDistribute.DUMA.data_process_base import DataProcessor, InputFeature_MultiChoice, InputFeatures

logger = logging.getLogger("Chn")


class DUMA_Processor(DataProcessor):
    def get_train_examples(self, args_in):
        input_file = os.path.join(args_in.train_path)
        logger.info(f"Load Chn train data from: [{input_file}]")
        return self._create_examples(
            records=self._read_csv(input_file),
            set_type='train'
        )

    # def get_dev_examples(self, data_dir, args): # dev已经处理好在train中
    #     input_file = os.path.join(data_dir,args)
    def get_test_examples(self, args):
        input_file = os.path.join(args.test_path)
        logger.info(f"Load Chn test data from: [{input_file}]")
        return self._create_examples(
            records=self._read_csv(input_file),
            set_type='test'
        )

    def _create_examples(self, records, set_type='train'):
        """为训练集和测试集创建示例"""
        examples = []

        for (i, line) in enumerate(records):
            record = line
            ex_id = str(i)
            guid = "%s-%s" % (set_type, ex_id)


class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):  # 将一条数据从(文章,问题,4个选项)转成(文章,问题,选项1)、(文章,问题,选项2)...
        label = self.df.label.values[idx]
        question = self.df.Question.values[idx]
        content = self.df.Content.values[idx]
        # [A、屏住呼吸, B、收敛行迹, C、退避迁徙, D、抑止打压]
        choice = self.df.Choices.values[idx][2:-2].split('\', \'')
        if len(choice) < 4:  # 如果选项不满四个，就补“不知道”
            for i in range(4 - len(choice)):
                choice.append('D．不知道')

        content = [content for i in range(len(choice))]
        pair = [question + ' ' + i[2:] for i in choice]

        return content, pair, label


# def convert_examples_to_features()

if __name__ == '__main__':
    args = init_arg_parser()
    processer = DUMA_Processor()
    processer.get_train_examples(args)
