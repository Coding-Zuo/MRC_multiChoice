# -*- coding:utf-8 -*-
import json
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random
import os
import time
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import BertTokenizer, RobertaTokenizer, RobertaModel, BertModel, BertConfig
from transformers import BertForMultipleChoice, RobertaForMultipleChoice
from transformers import get_cosine_schedule_with_warmup
from transformers import AdamW
import utils
import tokenizers
import transformers

parser = argparse.ArgumentParser(description='Haihua RC')
parser.add_argument('-m', '--model', default="/data2/roberta/RoBERTa_zh_L12",
                    help="model pretrain")
parser.add_argument('--data', metavar='DIR', default="/home/zuoyuhui/DataGame/haihuai_RC/data/", help="path to dataset")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=8, type=int, metavar='N', help="number of total epochs to run")
parser.add_argument('-tb', '--train_bs', default=16, metavar='N')
parser.add_argument('-vb', '--valid_bs', default=8, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=1e-4, metavar='LR', help='initial learning rate')
parser.add_argument('--max_len', default=256, type=float, help="max text len in bert")
parser.add_argument('--fold_num', default=2, type=int, metavar='N', help="jiaocha yanzheng")
parser.add_argument('--seed', default=42, type=int, metavar='N', help="random seed")
parser.add_argument('--accum_iter', default=2, type=int, metavar='N', help="gradient Accumulation")
parser.add_argument('--weight_decay', default=1e-4, type=float)

TOKENIZER = BertTokenizer.from_pretrained(parser.parse_args().model)


def collate_fn(data):  # 将文章问题选项拼在一起后，得到分词后的数字id，输出的size是(batch,n_choices,max_len
    input_ids, attention_mask, token_type_ids = [], [], []
    for x in data:
        text = TOKENIZER(x[1],
                         text_pair=x[0],
                         padding='max_length',  # 填充到使用参数max_length指定的最大长度，或者填充到模型的最大可接受输入长度(如果未提供该参数)。
                         truncation=True,
                         # TRUE或‘LIMEST_FIRST’：截断到使用参数max_length指定的最大长度，或者截断到模型的最大可接受输入长度(如果没有提供该参数)。这将逐个令牌截断令牌，如果提供了一对序列(或一批对)，则从该对中最长的序列中删除一个令牌。
                         max_length=parser.parse_args().max_len,
                         return_tensors='pt')  # 返回pytorch tensor格式
        input_ids.append(text['input_ids'].tolist())
        attention_mask.append(text['attention_mask'].tolist())
        token_type_ids.append(text['token_type_ids'].tolist())
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    label = torch.tensor([x[-1] for x in data])
    return input_ids, attention_mask, token_type_ids, label


class Robert_custom(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(Robert_custom, self).__init__(conf)
        self.Roberta = RobertaModel.from_pretrained(parser.parse_args().model, config=conf)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, _, hidden_state = self.Roberta(input_ids, token_type_ids, attention_mask)
        pass


def main(args):
    model_config = transformers.RobertaConfig.from_pretrained(args.model)
    model_config.output_hidden_states = True
    model = Robert_custom(conf=model_config).cuda()

    train_df = pd.read_csv(args.data + 'train_in.csv')
    folds = StratifiedKFold(n_splits=args.fold_num, shuffle=True, random_state=args.seed).split(
        np.arange(train_df.shape[0]), train_df.label.values)
    cv = []  # 保存每折的最佳准确率


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
