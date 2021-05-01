# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
import jsonlines
from tqdm import tqdm
import time
from sklearn.model_selection import StratifiedKFold
from transformers import BertTokenizer
from transformers import get_cosine_schedule_with_warmup
from transformers import AdamW

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import utils
import distribute_utils
import torch.distributed.launch
import torch.multiprocessing as mp
from AdversarialUtils import FGM, PGD, FreeLB
import AdversarialUtils
from bertBaseDistribute.LongFormer.args import init_arg_parser
from bertBaseDistribute.LongFormer.dataset.dataset_utils import *
from bertBaseDistribute.LongFormer.LongFormer import LongformerForMultipleChoice
from bertBaseDistribute.LongFormer.longformer.sliding_chunks import pad_to_window_size

args_parser = init_arg_parser()
pre_model = args_parser.longformer_cn_4096_base
tokenizer = BertTokenizer.from_pretrained(pre_model)


def main():
    utils.seed_everything(args_parser.seed)
    if args_parser.mult_gpu:
        mp.spawn(main_worker, nprocs=args_parser.nprocs, args=(args_parser.nprocs, args_parser))
    else:
        main_worker(args_parser.sigle_device, 1, args_parser)


def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank
    args.lr *= nprocs
    print("当前学习率：", args.lr)

    if args.mult_gpu:
        dist.init_process_group(backend=args.backend, init_method=args.tcp, world_size=args.nprocs,
                                rank=local_rank)
    torch.cuda.set_device(local_rank)

    data_set = QADataset(args, mode='train')
    train_data, train_label = data_set.get_data_labels()

    folds = StratifiedKFold(n_splits=args.kfold_num, shuffle=True, random_state=args.kfold_num).split(
        np.arange(len(train_data)), train_label)
    cv = []  # 保存每折的最佳准确率

    for fold, (train_idx, val_idx) in enumerate(folds):
        train = [train_data[i] for i in train_idx]
        val = [train_data[i] for i in train_idx]


class QADataset(Dataset):
    def __init__(self, args, mode='train', num_sample=None, mask_padding_with_zero=True):
        self.data, self.labels = self._get_train_or_test_data_list(args, mode)
        self.seqlen = args.max_len
        self._tokenizer = tokenizer

    def _get_train_or_test_data_list(self, args, mode):
        list = []
        labels = []
        file_path = args.train_path if mode == 'train' else args.test_path
        with open(file_path, mode='r') as json_file:
            reader = jsonlines.Reader(json_file)
            for instance in reader:
                article = instance['article'].strip().replace(' ', '').replace('\u3000', '') \
                    .replace('\n', '').replace('\xa0', '')
                question = instance['question'].strip().replace(' ', '').replace('\u3000', '') \
                    .replace('\n', '').replace('\xa0', '')
                opt1 = instance['option_0']
                opt1 = self.replace_placeholder(question, opt1)
                opt2 = instance['option_1']
                opt2 = self.replace_placeholder(question, opt2)
                opt3 = instance['option_2']
                opt3 = self.replace_placeholder(question, opt3)
                opt4 = instance['option_3']
                opt4 = self.replace_placeholder(question, opt4)

                list.append({
                    'text': article + '[SEP]' + question + '[SEP]' + opt1 + '[SEP]' + opt2 + '[SEP]' + opt3 + '[SEP]' + opt4,
                    'label': instance['label'],
                })
                if mode == "train":
                    labels.append(instance['label'])
        return list, labels

    def _convert_to_tensors(self, instance):
        def tok(s):
            return self._tokenizer.tokenize(s)


    def replace_placeholder(self, str, opt):
        list = str.split(' ')
        for i in range(len(list)):
            if list[i] == '（）':
                list[i] = opt
        final_opt = " ".join(list)
        return final_opt

    def __len__(self):
        return len(self.data)

    def get_data(self):
        return self.data

    def get_data_labels(self):
        return self.data, self.labels

    def __getitem__(self, item):
        return self._convert_to_tensors(item)




def main_process_call(str):
    if distribute_utils.is_main_process():
        print(str)


if __name__ == '__main__':
    main()
