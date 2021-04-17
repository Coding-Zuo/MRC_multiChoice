# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertForMultipleChoice
import utils
import distribute_utils
from args import init_arg_parser
import baselineMultGPU

args = init_arg_parser()

Param = {  # 训练的参数配置
    'fold_num': 5,  # 五折交叉验证
    'seed': 42,
    #     'model': 'hfl/chinese-bert-wwm-ext', #预训练模型
    'max_len': 256,  # 文本截断的最大长度
    'epochs': 8,
    'train_bs': 8,  # batch_size，可根据自己的显存调整
    'valid_bs': 8,
    'lr': 2e-5,  # 学习率
    'num_workers': 4,
    'accum_iter': 2,  # 梯度累积，相当于将batch_size*2
    'weight_decay': 1e-4,  # 权重衰减，防止过拟合
    'device': 0,
    'path_data': '/home/zuoyuhui/DataGame/haihuai_RC/data/',
    'model': '/home/zuoyuhui/DataGame/haihuai_RC/chinese-bert-wwm-ext',
}


def load_checkpoint(model, checkpoint_PATH, optimizer):
    model_CKPT = torch.load(checkpoint_PATH)
    model.load_state_dict(model_CKPT['state_dict'])
    optimizer.load_state_dict(model_CKPT['optimizer'])
    print('loading checkpoint!')
    return model, optimizer


def preprocessing_df():
    train_df = pd.read_csv(Param['path_data'] + 'train.csv')
    test_df = pd.read_csv(Param['path_data'] + 'test.csv')

    utils.seed_everything(Param['seed'])
    train_df['label'] = train_df['Answer'].apply(lambda x: ['A', 'B', 'C', 'D'].index(x))
    test_df['label'] = 0
    return train_df, test_df


def test2submit(test_df, model_state_name, model_num, model_name, output_name):
    device = torch.device('cuda')
    test_set = utils.MyDataset(test_df)
    test_loader = DataLoader(test_set, batch_size=Param['valid_bs'], collate_fn=baselineMultGPU.collate_fn,
                             shuffle=False,
                             num_workers=Param['num_workers'])

    model = BertForMultipleChoice.from_pretrained(Param['model']).to(device)
    predictions = []

    for fold in range(model_num):  # 把训练后的五个模型挨个进行预测
        y_pred = []
        model.load_state_dict(torch.load(model_state_name.format(model_name.split('/')[-1], fold)),
                              strict=False)
        print(model_name.split('/')[-1], fold)

        with torch.no_grad():
            tk = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
            for idx, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
                input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(
                    device), token_type_ids.to(device), y.to(device).long()

                output = model(input_ids, attention_mask, token_type_ids).logits.cpu().numpy()

                y_pred.extend(output)

        predictions += [y_pred]
    predictions = np.mean(predictions, 0).argmax(1)  # 将结果按五折进行平均，然后argmax得到label
    sub = pd.read_csv(Param['path_data'] + '/sample.csv', dtype=object)
    sub['label'] = predictions
    sub['label'] = sub['label'].apply(lambda x: ['A', 'B', 'C', 'D'][x])
    sub.to_csv(output_name, index=False)


# chinese-bert-wwm-ext_fold_0.pt
if __name__ == '__main__':
    train_df, test_df = preprocessing_df()
    # train_df.to_csv(args.data_dir + "train_label.csv", index=False)
    # train_df.to_csv(args.data_dir + "test_label.csv", index=False)
    # test2submit(test_df, model_state_name="spawn_adv_{}_fold_{}.pt", model_num=3, model_name="chinese-bert-wwm-ext",
    #             output_name="sub_spawn_adv_fold3.csv")

    """头两折是pgd后面两折是fgm"""
    # test2submit(test_df, model_state_name="spawn_adv_pgd_fgm_{}_fold_{}.pt", model_num=4,
    #             model_name="chinese-bert-wwm-ext",
    #             output_name="sub_spawn_adv_pgd_fgm_fold4.csv")
    # test2submit(test_df, model_state_name="spawn_adv_pgd_fgm_{}_fold_{}.pt", model_num=2,
    #             model_name="chinese-bert-wwm-ext",
    #             output_name="sub_spawn_adv_pgd_fold2.csv")

    # fgm bert 1e-5 single
    # test2submit(test_df, model_state_name="spawn_adv_pgd_{}_fold_{}.pt", model_num=5,
    #             model_name="chinese-bert-wwm-ext",
    #             output_name="sub_spawn_adv_fgm_fold5.csv")
    # test2submit(test_df, model_state_name="roformer/{}_fold_{}_roformer.pt", model_num=5,
    #             model_name="chinese_roformer_base",
    #             output_name="roformer_single_fold5.csv")
    # bert ema fgm 10epoch
    test2submit(test_df, model_state_name="train/robert_fgm_early_{}_fold_{}.pt", model_num=5,
                model_name="chinese-bert-wwm-ext",
                output_name="bert_ema_10ep_fold5.csv")
