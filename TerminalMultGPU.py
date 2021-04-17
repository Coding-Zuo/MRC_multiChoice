# -*- coding:utf-8 -*-
# python -m torch.distributed.launch --nproc_per_node=4 --use_env train_multi_gpu_using_launch.py
import os
import math
import tempfile
import argparse

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim

from tqdm import tqdm
import time
from sklearn.model_selection import StratifiedKFold
from transformers import BertTokenizer, AutoTokenizer
from transformers import BertForMultipleChoice, RobertaForMultipleChoice
from transformers import get_cosine_schedule_with_warmup
from transformers import AdamW

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import utils
import distribute_utils
import torch.distributed.launch

Param = {  # 训练的参数配置
    'fold_num': 5,  # 五折交叉验证
    'seed': 42,
    'max_len': 400,#256,  # 文本截断的最大长度
    'epochs': 8,
    'train_bs': 6,  # batch_size，可根据自己的显存调整
    'valid_bs': 6,
    'lr': 1e-5,  # 学习率
    'num_workers': 4,
    'accum_iter': 4,  # 梯度累积，相当于将batch_size*2
    'weight_decay': 1e-4,  # 权重衰减，防止过拟合
    'device': 0,
    'path_data': '/home/zuoyuhui/DataGame/haihuai_RC/data/',
    'model': '/home/zuoyuhui/DataGame/haihuai_RC/chinese-bert-wwm-ext',
}

tokenizer = AutoTokenizer.from_pretrained(Param['model'])


def preprocessing_df():
    train_df = pd.read_csv(Param['path_data'] + 'train.csv')
    test_df = pd.read_csv(Param['path_data'] + 'test.csv')

    utils.seed_everything(Param['seed'])
    train_df['label'] = train_df['Answer'].apply(lambda x: ['A', 'B', 'C', 'D'].index(x))
    test_df['label'] = 0
    return train_df, test_df


def collate_fn(data):  # 将文章问题选项拼在一起后，得到分词后的数字id，输出的size是(batch,n_choices,max_len
    input_ids, attention_mask, token_type_ids = [], [], []
    for x in data:
        text = tokenizer(x[1],
                         text_pair=x[0],
                         padding='max_length',  # 填充到使用参数max_length指定的最大长度，或者填充到模型的最大可接受输入长度(如果未提供该参数)。
                         truncation=True,
                         # TRUE或‘LIMEST_FIRST’：截断到使用参数max_length指定的最大长度，或者截断到模型的最大可接受输入长度(如果没有提供该参数)。这将逐个令牌截断令牌，如果提供了一对序列(或一批对)，则从该对中最长的序列中删除一个令牌。
                         max_length=Param['max_len'],
                         return_tensors='pt')  # 返回pytorch tensor格式
        input_ids.append(text['input_ids'].tolist())
        attention_mask.append(text['attention_mask'].tolist())
        token_type_ids.append(text['token_type_ids'].tolist())
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    label = torch.tensor([x[-1] for x in data])
    return input_ids, attention_mask, token_type_ids, label


def train_one_epoch(model, optimizer, train_loader, device, criterion, scheduler, scaler, epoch):
    model.train()

    losses = utils.AverageMeter1()
    accs = utils.AverageMeter1()

    optimizer.zero_grad()
    if distribute_utils.is_main_process():
        train_loader = tqdm(train_loader, total=len(train_loader), position=0, leave=True)

    mean_loss = torch.zeros(1).to(device)
    for step, (input_ids, attention_mask, token_type_ids, y) in enumerate(train_loader):
        input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(
            device), token_type_ids.to(device), y.to(device).long()

        with autocast():
            output = model(input_ids, attention_mask, token_type_ids).logits
            loss = criterion(output, y) / Param['accum_iter']
            # print(loss)
            scaler.scale(loss).backward()
            loss = distribute_utils.reduce_value(loss, average=True)
            mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

            if ((step + 1) % Param['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        if distribute_utils.is_main_process():
            acc = (output.argmax(1) == y).sum().item() / y.size(0)
            # acc = distribute_utils.reduce_value(torch.tensor(acc).to(device), average=True)
            accs.update(acc, y.size(0))
            losses.update(mean_loss * Param['accum_iter'], y.size(0))
            train_loader.set_postfix(loss=losses.avg, acc=accs.avg)
    return losses.avg, accs.avg


def eval_one_epoch(model, val_loader, criterion, device):
    model.eval()
    losses = utils.AverageMeter1()
    accs = utils.AverageMeter1()

    y_truth, y_pred = [], []
    sum_num = torch.zeros(1).to(device)
    with torch.no_grad():
        if distribute_utils.is_main_process():
            val_loader = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        for idx, (input_ids, attention_mask, token_type_ids, y) in enumerate(val_loader):
            input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(
                device), token_type_ids.to(device), y.to(device).long()
            output = model(input_ids, attention_mask, token_type_ids).logits
            y_truth.extend(y.cpu().numpy())
            y_pred.extend(output.argmax(1).cpu().numpy())
            loss = criterion(output, y)
            # print("output", output)
            # print("y", y)
            # print("y_pred", y_pred)
            # print("y_truth", y_truth)
            # print("output.argmax(1)", output.argmax(1))
            # print("output.argmax(1) == y", output.argmax(1) == y)
            # print("(output.argmax(1) == y).sum()", (output.argmax(1) == y).sum())
            # print("y.size(0)", y.size(0))
            sum_num += (output.argmax(1) == y).sum()
            acc = (output.argmax(1) == y).sum().item() / y.size(0)
            # print('---------------', acc)
            accs.update(acc, y.size(0))
            losses.update(loss.item(), y.size(0))
            if distribute_utils.is_main_process():
                val_loader.set_postfix(loss=losses.avg, acc=accs.avg)

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)
    sum_num = distribute_utils.reduce_value(sum_num, average=False)
    # val_loss = losses.avg
    # val_acc = accs.avg

    return sum_num


# class RCModel(nn.Module):
#     def __init__(self):
#         super(RCModel, self).__init__()
#         self.bert_model = RobertaForMultipleChoice.from_pretrained(Param['model'], )


def main(args, train_df):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    # 初始化
    distribute_utils.init_distributed_mode(args=args)
    device = torch.device(opt.device)
    rank = args.rank
    # Param['lr'] *= opt.world_size   # 并行时，梯度在多块gpu取均值，增大学习率
    Param['lr'] *= args.world_size  # 并行时，梯度在多块gpu取均值，增大学习率
    print("学习率变为：", Param['lr'])

    folds = StratifiedKFold(n_splits=Param['fold_num'], shuffle=True, random_state=Param['seed']).split(
        np.arange(train_df.shape[0]), train_df.label.values)
    cv = []  # 保存每折的最佳准确率

    for fold, (train_idx, val_idx) in enumerate(folds):
        train = train_df.loc[train_idx]
        val = train_df.loc[val_idx]

        train_set = utils.MyDataset(train)
        val_set = utils.MyDataset(val)

        # 给每个rank对应的进程分配训练的样本索引
        train_sampler = DistributedSampler(train_set)
        val_sampler = DistributedSampler(val_set)
        # 将样本索引每batch_size个元素组成一个list 验证集不用
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size=Param['train_bs'], drop_last=True)
        Param['num_workers'] = min(
            [os.cpu_count(), Param['train_bs'] if Param['train_bs'] > 1 else 0, 8])  # number of workers

        if rank == 0:
            print('Using {} dataloader workers every process'.format(Param['num_workers']))

        train_loader = DataLoader(train_set, batch_sampler=train_batch_sampler,
                                  pin_memory=True,
                                  collate_fn=collate_fn,
                                  num_workers=Param['num_workers'])
        val_loader = DataLoader(val_set, batch_size=Param['valid_bs'], sampler=val_sampler,
                                pin_memory=True,
                                collate_fn=collate_fn,
                                num_workers=Param['num_workers'])
        best_acc = 0

        #  注意：训练时要保证每个gpu初始权重一模一样
        model = BertForMultipleChoice.from_pretrained(Param['model']).to(device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # 转为DDP模型
            # model = DistributedDataParallel(model, device_ids=[args.gpu])
            model = DistributedDataParallel(model, device_ids=[rank], output_device=0)

        """
        优化配置
        """
        scaler = GradScaler()
        optimizer = AdamW(model.parameters(), lr=Param['lr'], weight_decay=Param['weight_decay'])
        criterion = nn.CrossEntropyLoss().to(device)
        scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // Param['accum_iter'],
                                                    Param['epochs'] * len(train_loader) // Param['accum_iter'])

        for epoch in range(Param['epochs']):
            print('epochs:', epoch)
            # 通过epoch 设置seed生成器初始化种子 打乱数据顺序结果
            train_sampler.set_epoch(epoch)
            time.sleep(0.2)

            # train
            train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device, criterion, scheduler,
                                                    scaler, epoch)

            # val
            # val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
            sum_num = eval_one_epoch(model, val_loader, criterion, device)
            acc = sum_num / val_sampler.total_size

            if acc > best_acc:
                best_acc = acc
                print("best:", best_acc)
                if rank == 0:
                    torch.save(model.module.state_dict(),
                               '{}_fold_{}_3gpu_max400.pt'.format(Param['model'].split('/')[-1], fold))
                    # torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
                    #             'optimizer': optimizer.state_dict(), 'alpha': loss.alpha, 'gamma': loss.gamma},
                    #            checkpoint_path + '/m-' + launchTimestamp + '-' + str("%.4f" % lossMIN) + '.pth.tar')

        cv.append(best_acc)
        print("cv:", np.mean(cv))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--nproc_per_node', default='4')
    parser.add_argument('--use_env')
    opt = parser.parse_args()
    """
    初始环境设置
    """
    print("Let's use", opt.world_size, "GPUs!")

    """
    预处理数据
    """
    train_df, test_df = preprocessing_df()
    """
    开始训练
    """
    main(opt, train_df)

    # test2submit(test_df)

    distribute_utils.cleanup()
