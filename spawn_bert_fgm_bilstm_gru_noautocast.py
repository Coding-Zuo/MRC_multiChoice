# -*- coding:utf-8 -*-
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
from transformers import BertForMultipleChoice, RobertaForMultipleChoice, RobertaModel
from transformers import get_cosine_schedule_with_warmup
from transformers import AdamW

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import utils
import distribute_utils
import torch.distributed.launch
import torch.multiprocessing as mp
import BertModelsCustom
from AdversarialUtils import FGM, PGD
from pytorchtools import EarlyStopping

# https://github.com/tczhangzhi/pytorch-distributed/blob/master/multiprocessing_distributed.py
# https://zhuanlan.zhihu.com/p/98535650
parser = argparse.ArgumentParser(description='Haihua RC')
parser.add_argument('-m', '--model', default="/home/zuoyuhui/DataGame/haihuai_RC/chinese-bert-wwm-ext",
                    help="model pretrain")
parser.add_argument('--data', metavar='DIR', default="/home/zuoyuhui/DataGame/haihuai_RC/data/", help="path to dataset")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=8, type=int, metavar='N', help="number of total epochs to run")
parser.add_argument('-b', '--batch_size', default=8, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=1e-5, metavar='LR', help='initial learning rate')
parser.add_argument('--max_len', default=256, type=float, help="max text len in bert")
parser.add_argument('--fold_num', default=5, type=int, metavar='N', help="jiaocha yanzheng")
parser.add_argument('--seed', default=42, type=int, metavar='N', help="random seed")
parser.add_argument('--accum_iter', default=2, type=int, metavar='N', help="gradient Accumulation")
parser.add_argument('--weight_decay', default=1e-4, type=float)

tokenizer = BertTokenizer.from_pretrained(parser.parse_args().model)


def collate_fn(data):  # ???????????????????????????????????????????????????????????????id????????????size???(batch,n_choices,max_len
    input_ids, attention_mask, token_type_ids = [], [], []
    for x in data:
        text = tokenizer(x[1],
                         text_pair=x[0],
                         padding='max_length',  # ?????????????????????max_length???????????????????????????????????????????????????????????????????????????(????????????????????????)???
                         truncation=True,
                         # TRUE??????LIMEST_FIRST???????????????????????????max_length???????????????????????????????????????????????????????????????????????????(???????????????????????????)???????????????????????????????????????????????????????????????(????????????)?????????????????????????????????????????????????????????
                         max_length=parser.parse_args().max_len,
                         return_tensors='pt')  # ??????pytorch tensor??????
        input_ids.append(text['input_ids'].tolist())
        attention_mask.append(text['attention_mask'].tolist())
        token_type_ids.append(text['token_type_ids'].tolist())
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    label = torch.tensor([x[-1] for x in data])
    return input_ids, attention_mask, token_type_ids, label


def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count() - 3
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))


def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank
    args.lr *= nprocs
    print(args.lr)
    utils.seed_everything(args.seed)
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=args.nprocs,
                            rank=local_rank)
    torch.cuda.set_device(local_rank)

    train_df = pd.read_csv(args.data + 'train_in.csv')
    folds = StratifiedKFold(n_splits=args.fold_num, shuffle=True, random_state=args.seed).split(
        np.arange(train_df.shape[0]), train_df.label.values)
    cv = []  # ??????????????????????????????

    for fold, (train_idx, val_idx) in enumerate(folds):
        train = train_df.loc[train_idx]
        val = train_df.loc[val_idx]
        train_set = utils.MyDataset(train)
        val_set = utils.MyDataset(val)

        model = BertModelsCustom.BertForMultipleChoice.from_pretrained(args.model).cuda(local_rank)
        # args.batch_size = int(args.batch_size / args.nprocs)
        model = DistributedDataParallel(model, device_ids=[local_rank])

        train_sampler = DistributedSampler(train_set)
        val_sampler = DistributedSampler(val_set)

        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size=args.batch_size, drop_last=True)
        train_loader = DataLoader(train_set, batch_sampler=train_batch_sampler, pin_memory=False,
                                  collate_fn=collate_fn, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=val_sampler, pin_memory=False,
                                collate_fn=collate_fn, num_workers=2)

        best_acc = 0

        scaler = GradScaler()
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss().cuda(local_rank)
        scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // args.accum_iter,
                                                    args.epochs * len(train_loader) // args.accum_iter)

        if args.use_early_stop:
            early_stopping = EarlyStopping(patience=args.early_stop_num)

        for epoch in range(args.epochs):
            print('epochs:', epoch)
            time.sleep(0.02)
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

            train_one_epoch(train_loader, model, criterion, optimizer, scheduler, local_rank, scaler, args)

            val_loss, val_acc = eval_one_epoch(val_loader, model, criterion, local_rank, args)

            if args.use_early_stop:
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    break

            if val_acc > best_acc:
                best_acc = val_acc
                print("best:", best_acc)
                if distribute_utils.is_main_process():
                    torch.save(model.module.state_dict(),
                               'spawn_single_fgm_early_{}_fold_{}.pt'.format(args.model.split('/')[-1], fold))

        cv.append(best_acc)
        if distribute_utils.is_main_process():
            print("cv:", np.mean(cv))


def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, local_rank, scaler, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    accs = utils.AverageMeter('Acc', ':6.2f')
    model.train()

    end = time.time()
    optimizer.zero_grad()
    if distribute_utils.is_main_process():
        train_loader = tqdm(train_loader, total=len(train_loader), position=0, leave=True)

    fgm = FGM(model)
    pgd = PGD(model)
    K = 3
    y_truth, y_pred = [], []
    mean_loss = torch.zeros(1).cuda(local_rank)
    for step, (input_ids, attention_mask, token_type_ids, y) in enumerate(train_loader):
        input_ids, attention_mask, token_type_ids, y = map(lambda x: x.cuda(local_rank, non_blocking=True),
                                                           (input_ids, attention_mask, token_type_ids, y))
        data_time.update(time.time() - end)
        with autocast():
            output = model(input_ids, attention_mask, token_type_ids)[0]
            loss = criterion(output, y) / args.accum_iter
            scaler.scale(loss).backward()

            # ????????????
            fgm.attack()
            output_adv = model(input_ids, attention_mask, token_type_ids)[0]
            loss_adv = criterion(output_adv, y) / args.accum_iter
            del input_ids, attention_mask, token_type_ids
            loss_adv.backward()  # ??????????????????????????????grad???????????????????????????????????????
            fgm.restore()  # ??????embedding??????

            # pgd.backup_grad()
            # ????????????
            # for t in range(K):
            #     pgd.attack(is_first_attack=(t == 0))  # ???embedding?????????????????????, first attack?????????param.data
            #     if t != K - 1:
            #         model.zero_grad()
            #     else:
            #         pgd.restore_grad()
            #     output_adv = model(input_ids, attention_mask, token_type_ids)[0]
            #     loss_adv = criterion(output_adv, y) / args.accum_iter
            #     loss_adv.backward()  # ??????????????????????????????grad???????????????????????????????????????
            # pgd.restore()  # ??????embedding??????

            loss = distribute_utils.reduce_value(loss, average=True)
            mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

            if ((step + 1) % args.accum_iter == 0) or ((step + 1) == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        if distribute_utils.is_main_process():
            acc = (output.argmax(1) == y).sum().item() / y.size(0)
            accs.update(acc, y.size(0))
            losses.update(mean_loss.item() * args.accum_iter, y.size(0))
            train_loader.set_postfix(loss=losses.avg, acc=accs.avg)

    return losses.avg, accs.avg


@torch.no_grad()
def eval_one_epoch(val_loader, model, criterion, local_rank, args):
    losses = utils.AverageMeter('Loss', ':.4e')
    accs = utils.AverageMeter('Acc', ':6.2f')
    model.eval()

    end = time.time()
    y_truth, y_pred = [], []
    if distribute_utils.is_main_process():
        val_loader = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
    for idx, (input_ids, attention_mask, token_type_ids, y) in enumerate(val_loader):
        input_ids, attention_mask, token_type_ids, y = input_ids.cuda(local_rank,
                                                                      non_blocking=True), attention_mask.cuda(
            local_rank, non_blocking=True), token_type_ids.cuda(local_rank, non_blocking=True), y.cuda(local_rank,
                                                                                                       non_blocking=True).long()
        output = model(input_ids, attention_mask, token_type_ids)[0]
        y_truth.extend(y.cpu().numpy())
        y_pred.extend(output.argmax(1).cpu().numpy())
        loss = criterion(output, y)
        dist.barrier()
        acc = (output.argmax(1) == y).sum().item() / y.size(0)

        reduced_loss = distribute_utils.reduce_mean(loss, args.nprocs)

        if distribute_utils.is_main_process():
            accs.update(acc, y.size(0))
            losses.update(reduced_loss.item(), y.size(0))
            val_loader.set_postfix(loss=losses.avg, acc=accs.avg)

        # ??????????????????????????????
        if torch.device(local_rank) != torch.device("cpu"):
            torch.cuda.synchronize(local_rank)

    return losses.avg, accs.avg


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
