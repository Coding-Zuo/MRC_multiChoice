# -*- coding:utf-8 -*-
import json
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
import BertModelsCustom
import utils

Param = {  # 训练的参数配置
    'fold_num': 5,  # 五折交叉验证
    'seed': 42,
    'model': '/data2/roberta/RoBERTa_zh_L12',
    'path_data': '/home/zuoyuhui/DataGame/haihuai_RC/data/',
    'max_len': 256,  # 文本截断的最大长度
    'epochs': 8,
    'train_bs': 16,  # batch_size，可根据自己的显存调整
    'valid_bs': 16,
    'lr': 1e-5,  # 学习率
    'num_workers': 2,
    'accum_iter': 2,  # 梯度累积，相当于将batch_size*2
    'weight_decay': 1e-5,  # 权重衰减，防止过拟合
    'device': 0,
}

tokenizer = BertTokenizer.from_pretrained(Param['model'])


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


def main():
    utils.seed_everything(Param['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_df = pd.read_csv(Param['path_data'] + 'train.csv')
    test_df = pd.read_csv(Param['path_data'] + 'test.csv')
    train_df['label'] = train_df['Answer'].apply(lambda x: ['A', 'B', 'C', 'D'].index(x))
    test_df['label'] = 0

    model = BertModelsCustom.RobertaForMultipleChoice.from_pretrained(Param['model']).to(device)

    folds = StratifiedKFold(n_splits=Param['fold_num'], shuffle=True, random_state=Param['seed']) \
        .split(np.arange(train_df.shape[0]), train_df.label.values)  # 五折交叉验证

    cv = []  # 保存每折的最佳准确率

    for fold, (trn_idx, val_idx) in enumerate(folds):
        train = train_df.loc[trn_idx]
        val = train_df.loc[val_idx]

        train_set = utils.MyDataset(train)
        val_set = utils.MyDataset(val)

        train_loader = DataLoader(train_set, batch_size=Param['train_bs'], collate_fn=collate_fn, shuffle=True,
                                  num_workers=Param['num_workers'])
        val_loader = DataLoader(val_set, batch_size=Param['valid_bs'], collate_fn=collate_fn, shuffle=False,
                                num_workers=Param['num_workers'])

        best_acc = 0

        scaler = GradScaler()
        optimizer = AdamW(model.parameters(), lr=Param['lr'], weight_decay=Param['weight_decay'])  # AdamW优化器
        criterion = nn.CrossEntropyLoss()
        scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // Param['accum_iter'],
                                                    Param['epochs'] * len(train_loader) // Param['accum_iter'])

        for epoch in range(Param['epochs']):

            print('epoch:', epoch)
            time.sleep(0.2)

            # train_loss, train_acc = train_model(model, train_loader)
            losses = utils.AverageMeter1()
            accs = utils.AverageMeter1()

            optimizer.zero_grad()
            tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)

            for step, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
                input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(
                    device), token_type_ids.to(device), y.to(device).long()
                # https://zhuanlan.zhihu.com/p/165152789
                with autocast():  # 使用半精度训练
                    output = model(input_ids, attention_mask, token_type_ids)
                    output = output.logits

                    loss = criterion(output, y) / Param['accum_iter']
                    scaler.scale(loss).backward()

                    if ((step + 1) % Param['accum_iter'] == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()

                acc = (output.argmax(1) == y).sum().item() / y.size(0)
                losses.update(loss.item() * Param['accum_iter'], y.size(0))
                accs.update(acc, y.size(0))

                tk.set_postfix(loss=losses.avg, acc=accs.avg)

            # val_loss, val_acc = test_model(model, val_loader)
            model.eval()

            losses = utils.AverageMeter1()
            accs = utils.AverageMeter1()

            y_truth, y_pred = [], []
            with torch.no_grad():
                tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
                for idx, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
                    input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(
                        device), token_type_ids.to(device), y.to(device).long()

                    output = model(input_ids, attention_mask, token_type_ids).logits

                    y_truth.extend(y.cpu().numpy())
                    y_pred.extend(output.argmax(1).cpu().numpy())

                    loss = criterion(output, y)
                    acc = (output.argmax(1) == y).sum().item() / y.size(0)

                    losses.update(loss.item(), y.size(0))
                    accs.update(acc, y.size(0))

                    tk.set_postfix(loss=losses.avg, acc=accs.avg)
            val_acc = accs.avg
            val_loss = losses.avg

            if val_acc > best_acc:
                best_acc = val_acc
                print(val_acc)
                torch.save(model.state_dict(),
                           '{}_fold_{}_robert_single.pt'.format(Param['model'].split('/')[-1], fold))

        cv.append(best_acc)
        print("cv:", np.mean(cv))


if __name__ == '__main__':
    main()
