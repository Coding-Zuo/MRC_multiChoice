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
from transformers import BertTokenizer, AutoTokenizer, BertConfig
from transformers import BertForMultipleChoice, RobertaForMultipleChoice, RobertaModel
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import AdamW

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import utils
import distribute_utils
import torch.distributed.launch
import torch.multiprocessing as mp
import BertModelsCustom
from AdversarialUtils import FGM, PGD, FreeLB
import AdversarialUtils
from args import init_arg_parser
from pytorchtools import EMA, EarlyStopping
import torchsummary
from operator import itemgetter
from pympler import tracker

args_parser = init_arg_parser()
pre_model = args_parser.bert_chinese_wwm_ext
tokenizer = BertTokenizer.from_pretrained(pre_model)


def collate_fn(data):  # 将文章问题选项拼在一起后，得到分词后的数字id，输出的size是(batch,n_choices,max_len
    input_ids, attention_mask, token_type_ids = [], [], []
    for x in data:
        text = tokenizer(x[1],
                         text_pair=x[0],
                         padding='max_length',  # 填充到使用参数max_length指定的最大长度，或者填充到模型的最大可接受输入长度(如果未提供该参数)。
                         truncation=True,
                         # TRUE或‘LIMEST_FIRST’：截断到使用参数max_length指定的最大长度，或者截断到模型的最大可接受输入长度(如果没有提供该参数)。这将逐个令牌截断令牌，如果提供了一对序列(或一批对)，则从该对中最长的序列中删除一个令牌。
                         max_length=args_parser.max_len,
                         return_tensors='pt')  # 返回pytorch tensor格式
        input_ids.append(text['input_ids'].tolist())
        attention_mask.append(text['attention_mask'].tolist())
        token_type_ids.append(text['token_type_ids'].tolist())
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    label = torch.tensor([x[-1] for x in data])
    return input_ids, attention_mask, token_type_ids, label


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

    train_df = pd.read_csv(args.train_path)

    folds = StratifiedKFold(n_splits=args.kfold_num, shuffle=True, random_state=args.seed).split(
        np.arange(train_df.shape[0]), train_df.label.values)
    cv = []  # 保存每折的最佳准确率

    for fold, (train_idx, val_idx) in enumerate(folds):
        train = train_df.loc[train_idx]
        val = train_df.loc[val_idx]
        train_set = utils.MyDataset(train)
        val_set = utils.MyDataset(val)

        bert_config = BertConfig.from_pretrained(pre_model, output_hidden_states=True)
        model = BertModelsCustom.BertForMultipleChoice(config=bert_config).from_pretrained(pre_model).cuda(
            local_rank)

        # get_info_param(model)
        # ema = EMA(model, decay=0.999)
        # ema.register()
        early_stopping = EarlyStopping(patience=2)

        if args.mult_gpu:
            model = DistributedDataParallel(model, device_ids=[local_rank])

            train_sampler = DistributedSampler(train_set)
            val_sampler = DistributedSampler(val_set)

            train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size=args.train_bs,
                                                                drop_last=True)
            train_loader = DataLoader(train_set, batch_sampler=train_batch_sampler, pin_memory=False,
                                      collate_fn=collate_fn, num_workers=2)
            val_loader = DataLoader(val_set, batch_size=args.valid_bs, sampler=val_sampler, pin_memory=False,
                                    collate_fn=collate_fn, num_workers=2)
        else:
            train_loader = DataLoader(train_set, batch_size=args.train_bs, collate_fn=collate_fn, shuffle=True,
                                      num_workers=args.num_workers)
            val_loader = DataLoader(val_set, batch_size=args.valid_bs, collate_fn=collate_fn, shuffle=False,
                                    num_workers=args.num_workers)

        best_acc = 0

        scaler = GradScaler()
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss().cuda(local_rank)

        # print(len(train_loader))
        # print(len(train_loader) / args.accum_iter)
        # print(args.epochs * len(train_loader) // args.accum_iter)
        scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // args.accum_iter,
                                                    args.epochs * len(train_loader) // args.accum_iter)
        # scheduler = get_cosine_schedule_with_warmup(optimizer, 100,
        #                                             args.epochs * len(train_loader) // args.accum_iter)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=args.warmup_steps,
        #     num_training_steps=args.epochs * len(train_loader) // args.accum_iter
        # )
        early_flag = 0
        for epoch in range(args.epochs):
            main_process_call('{} {}'.format('epoch：', epoch))
            time.sleep(1)

            if args.mult_gpu:
                train_sampler.set_epoch(epoch)
                val_sampler.set_epoch(epoch)

            train_one_epoch(train_loader, model, criterion, optimizer, scheduler, local_rank, scaler, args)
            val_loss, val_acc = eval_one_epoch(val_loader, model, criterion, local_rank, args)

            if args.use_early_stop and distribute_utils.is_main_process():
                dist.barrier()
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    early_flag = 1
                    main_process_call("----------------------------------------")
                    main_process_call("break this fold:gpu-" + str(local_rank))
                    main_process_call("----------------------------------------")
                    break
            if early_flag == 1:
                main_process_call("----------------------------------------")
                main_process_call("break this fold:gpu-" + str(local_rank))
                main_process_call("----------------------------------------")
                break

            if val_acc > best_acc:
                best_acc = val_acc
                print("(*^▽^*)(*^▽^*)(*^▽^*)小枪枪为我加油(*^▽^*)(*^▽^*)(*^▽^*)best:", best_acc)
                if args.mult_gpu:
                    if distribute_utils.is_main_process():
                        torch.save(model.module.state_dict(),
                                   args.save_model_name.format(pre_model.split('/')[-1], fold))
                else:
                    torch.save(model.module.state_dict(),
                               args.save_model_name.format(pre_model.split('/')[-1], fold))
        cv.append(best_acc)

        if args.mult_gpu:
            if distribute_utils.is_main_process():
                print("cv:", np.mean(cv))
        else:
            print("cv:", np.mean(cv))


def main_process_call(str):
    if distribute_utils.is_main_process():
        print(str)


def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, local_rank, scaler, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    accs = utils.AverageMeter('Acc', ':6.2f')

    get_info_param(model)
    model.train()

    end = time.time()
    optimizer.zero_grad()

    if args.mult_gpu:
        if distribute_utils.is_main_process():
            train_loader = tqdm(train_loader, total=len(train_loader), position=0, leave=True)
    else:
        train_loader = tqdm(train_loader, total=len(train_loader), position=0, leave=True)

    if args.use_fgm:
        fgm = FGM(model)
        main_process_call("使用fgm攻击")
    if args.use_pgd:
        pgd = PGD(model)
        main_process_call("使用pgd攻击")
    if args.use_FreeLB:
        freeLB = FreeLB(adv_K=args.k_pdg, adv_lr=1e-2, adv_init_mag=2, adv_max_norm=1)
        main_process_call("使用freeLB攻击")

    y_truth, y_pred = [], []
    mean_loss = torch.zeros(1).cuda(local_rank)
    for step, (input_ids, attention_mask, token_type_ids, y) in enumerate(train_loader):
        input_ids, attention_mask, token_type_ids, y = map(lambda x: x.cuda(local_rank, non_blocking=True),
                                                           (input_ids, attention_mask, token_type_ids, y))
        # [batch,4,256]
        data_time.update(time.time() - end)
        # [batch,4]
        output = model(input_ids, attention_mask, token_type_ids)[0]

        loss = criterion(output, y) / args.accum_iter
        scaler.scale(loss).backward()

        if args.use_fgm:
            AdversarialUtils.fgm_use_bert_adv(fgm, model, input_ids, attention_mask, token_type_ids, y, criterion,
                                              args)
        if args.use_pgd:
            AdversarialUtils.pgd_use_bert_adv(pgd, model, input_ids, attention_mask, token_type_ids, y, criterion,
                                              args)

        if args.use_FreeLB:
            AdversarialUtils.freeLB_use_bert_adv(freeLB, model, input_ids, attention_mask, token_type_ids, y,
                                                 criterion, args)

        del input_ids, attention_mask, token_type_ids

        loss = distribute_utils.reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

        if ((step + 1) % args.accum_iter == 0) or ((step + 1) == len(train_loader)):
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.step()
            scheduler.step()
            # ema.update()
            optimizer.zero_grad()

        if distribute_utils.is_main_process():
            acc = (output.argmax(1) == y).sum().item() / y.size(0)
            accs.update(acc, y.size(0))
            losses.update(mean_loss.item() * args.accum_iter, y.size(0))
            train_loader.set_postfix(loss=losses.avg, acc=accs.avg)
            batch_time.update(time.time() - end)
            end = time.time()

    return losses.avg, accs.avg


@torch.no_grad()
def eval_one_epoch(val_loader, model, criterion, local_rank, args):
    losses = utils.AverageMeter('Loss', ':.4e')
    accs = utils.AverageMeter('Acc', ':6.2f')
    model.eval()
    # ema.apply_shadow()

    end = time.time()
    y_truth, y_pred = [], []
    if args.mult_gpu:
        if distribute_utils.is_main_process():
            val_loader = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
    else:
        val_loader = tqdm(val_loader, total=len(val_loader), position=0, leave=True)

    for idx, (input_ids, attention_mask, token_type_ids, y) in enumerate(val_loader):
        input_ids, attention_mask, token_type_ids, y = input_ids.cuda(local_rank,
                                                                      non_blocking=True), attention_mask.cuda(
            local_rank, non_blocking=True), token_type_ids.cuda(local_rank, non_blocking=True), y.cuda(local_rank,
                                                                                                       non_blocking=True).long()
        output = model(input_ids, attention_mask, token_type_ids)[0]
        del input_ids, attention_mask, token_type_ids

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

        # 等待所有进程计算完毕
        if torch.device(local_rank) != torch.device("cpu"):
            torch.cuda.synchronize(local_rank)
    # ema.restore()
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


def get_info_param(model):
    if distribute_utils.is_main_process():
        # 定义总参数量、可训练参数量及非可训练参数量变量
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0

        # 遍历model.parameters()返回的全局参数列表
        for param in model.parameters():
            mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
            Total_params += mulValue  # 总参数量
            if param.requires_grad:
                Trainable_params += mulValue  # 可训练参数量
            else:
                NonTrainable_params += mulValue  # 非可训练参数量

        print(f'Total params: {Total_params}')
        print(f'Trainable params: {Trainable_params}')
        print(f'Non-trainable params: {NonTrainable_params}')


if __name__ == '__main__':
    main()
