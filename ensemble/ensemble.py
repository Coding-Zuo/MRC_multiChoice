# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import torch
import os

from sklearn.base import ClassifierMixin
from sklearn.ensemble._voting import _BaseVoting
from sklearn.utils import _deprecate_positional_args
from sklearn.utils.multiclass import check_classification_targets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertForMultipleChoice
from bertBaseDistribute import BertModelsCustom
import utils
import distribute_utils
from collections import Counter
from args import init_arg_parser
import TerminalMultGPU
from sklearn.ensemble import VotingClassifier


def preprocessing_df(args):
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    utils.seed_everything(args.seed)
    train_df['label'] = train_df['Answer'].apply(lambda x: ['A', 'B', 'C', 'D'].index(x))
    test_df['label'] = 0
    return train_df, test_df


def load_checkpoint(args):
    L = []
    for root, dirs, files in os.walk("/data2/code/bertBaseDistribute/models/ensemble_select"):
        for file in files:
            if os.path.splitext(file)[1] == '.pt':
                L.append(os.path.join(root, file))
    return L


@torch.no_grad()
def ensembel(test_df, check_point_list, model_list, output_name, args, strategy="argmax"):
    device = torch.device(3)
    test_set = utils.MyDataset(test_df)
    test_loader = DataLoader(test_set, batch_size=args.valid_bs, collate_fn=TerminalMultGPU.collate_fn,
                             shuffle=False,
                             num_workers=args.num_workers)

    predict = ""
    if strategy == "vote":
        predict = strategy_vote(checkpoint_list, model_list, test_loader, device)
    elif strategy == "stacking":
        pass
    else:
        predict = strategy_argmax(check_point_list, model_list, test_loader, device)

    submit2sample(args, predict, output_name)


def strategy_argmax(check_point_list, model_list, test_loader, device):
    predictions = []
    for check_poient in check_point_list:
        print(check_poient)
        if check_poient.find("linear3") >= 0:
            model = model_list["bert_3linear_model"]
        else:
            model = model_list["bert_normal"]

        y_pred = []
        model.load_state_dict(torch.load(check_poient), strict=False)

        tk = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
        for idx, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
            input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(
                device), token_type_ids.to(device), y.to(device).long()

            output = model(input_ids, attention_mask, token_type_ids)
            output = output[0].cpu().numpy()

            y_pred.extend(output)

        predictions += [y_pred]
    predict = np.mean(predictions, 0).argmax(1)  # 将结果按五折进行平均，然后argmax得到label
    return predict


def strategy_vote(check_point_list, model_list, test_loader, device):
    model_nets = []
    for check_poient in check_point_list:
        print(check_poient)
        if check_poient.find("linear3") >= 0:
            model = model_list["bert_3linear_model"]
        else:
            model = model_list["bert_normal"]
        model.load_state_dict(torch.load(check_poient), strict=False)
        model_nets.append(model)

    predict = []
    vclf = Voting(estimators=model_nets, voting='hard')

    tk = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
    for idx, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
        input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(
            device), token_type_ids.to(device), y.to(device).long()
        output = vclf.forward(input_ids, attention_mask, token_type_ids)
        predict.extend(output)
    return predict


class Voting:
    def __init__(self, estimators, *, voting='hard'):
        self.estimators = estimators
        self.voting = voting

    def forward(self, input_ids, attention_mask, token_type_ids):
        y_pred = []
        for model in self.estimators:
            output = model(input_ids, attention_mask, token_type_ids)
            output = output[0].cpu().numpy()
            output = np.argmax(output, 1)
            y_pred += [output]
        print(y_pred)
        y_pred = np.array(y_pred)
        pridict = []
        for i in range(len(y_pred[0])):
            pridict.append(Counter(y_pred[:, i]).most_common(1)[0][0])
        return pridict


def submit2sample(args, predict, output_name):
    if predict == "":
        print("没有策略")
        return
    sub = pd.read_csv(args.data_dir + '/sample.csv', dtype=object)
    sub['label'] = predict
    sub['label'] = sub['label'].apply(lambda x: ['A', 'B', 'C', 'D'][x])
    sub.to_csv(output_name, index=False)
    print("finish")


# chinese-bert-wwm-ext_fold_0.pt
if __name__ == '__main__':
    args = init_arg_parser()
    train_df, test_df = preprocessing_df(args)
    checkpoint_list = load_checkpoint(args)
    print(len(checkpoint_list))
    print(checkpoint_list)

    device = torch.device(3)
    bert_3linear_model = BertModelsCustom.BertForMultipleChoice3Linear.from_pretrained(args.bert_chinese_wwm_ext)
    bert_normal = BertModelsCustom.BertForMultipleChoice.from_pretrained(args.bert_chinese_wwm_ext)

    models = {
        "bert_3linear_model": bert_3linear_model.to(device),
        "bert_normal": bert_normal.to(device)
    }
    ensembel(test_df, checkpoint_list, models, "ensemble_final.csv", args, strategy="argmax")
    # ensembel(test_df, checkpoint_list, models, "ensemble_select1_vote.csv", args, strategy="vote")
