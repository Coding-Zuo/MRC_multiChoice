# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import os
import re
import jieba
import json
import torch
import pickle
from args import init_arg_parser
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import EnglishStemmer


def main(args):
    cut_pre_data(args)


def read_cut(args):
    train_df = pd.read_csv(args.data_dir + "train_cut.csv",
                           usecols=['cut_content', 'cut_Question', 'cut_Choice', 'label'])
    test_df = pd.read_csv(args.data_dir + "train_cut.csv",
                          usecols=['cut_content', 'cut_Question', 'cut_Choice', 'label'])
    return train_df, test_df


def processing(args):
    train_df = read_cut(args)
    test_df = read_cut(args)


def cut_pre_data(args):
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    stop_words_list = []
    with open(args.stopwords_path, 'r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            lline = line.strip()
            stop_words_list.append(lline)
    train_df_cut = pd.DataFrame()
    test_df_cut = pd.DataFrame()
    train_df_cut['label'] = train_df['label']
    test_df_cut['label'] = test_df['label']

    train_df_cut['cut_Choice'] = train_df.Choices.apply(chinese_pre, args=[stop_words_list, True])
    train_df_cut['cut_content'] = train_df.Content.apply(chinese_pre, args=[stop_words_list])
    train_df_cut['cut_Question'] = train_df.Question.apply(chinese_pre, args=[stop_words_list])

    test_df_cut['cut_content'] = train_df.Content.apply(chinese_pre, args=[stop_words_list, True])
    test_df_cut['cut_Question'] = train_df.Question.apply(chinese_pre, args=[stop_words_list])
    test_df_cut['cut_Choice'] = train_df.Choices.apply(chinese_pre, args=[stop_words_list])

    train_df_cut.to_csv(args.data_dir + "/train_cut.csv", index=False)
    test_df_cut.to_csv(args.data_dir + "/test_cut.csv", index=False)


# 中文 去除不需要的字符、分词 去停用词
def chinese_pre(text_data, stop_words, is_choice=False):
    # 字母转化为小写
    text_data = text_data.lower()
    text_data = re.sub("\d+", "", text_data)
    punctuation = '[\s+\.\!\/\_,$%^&*\\\(\)\（\）\'\"”“，。：．]+'
    if not is_choice:
        text_data = re.sub(punctuation, "", text_data)
    # 分词用精确模式
    text_data = list(jieba.cut(text_data, cut_all=False))
    # 去停用词和多余空格
    text_data = [word.strip() for word in text_data if word not in stop_words]
    # 处理后使用空格链接为字符串
    text_data = " ".join(text_data)
    return text_data


def dict2pickle(your_dict, out_file):
    with open(out_file, 'wb') as f:
        pickle.dump(your_dict, f)


def get_numpy_word_embed(word2ix):
    row = 0
    file = 'zhs_wiki_glove.vectors.100d.txt'
    path = '/home/socialbird/platform/aion-autonlp/Downloads'
    whole = os.path.join(path, file)
    words_embed = {}
    with open(whole, mode='r')as f:
        lines = f.readlines()
        for line in lines:
            # print(line)
            # print(len(line.split()))
            line_list = line.split()
            word = line_list[0]
            embed = line_list[1:]
            embed = [float(num) for num in embed]
            words_embed[word] = embed
            if row > 20000:
                break
            row += 1
    # word2ix = {}
    ix2word = {ix: w for w, ix in word2ix.items()}
    id2emb = {}
    for ix in range(len(word2ix)):
        if ix2word[ix] in words_embed:
            id2emb[ix] = words_embed[ix2word[ix]]
        else:
            id2emb[ix] = [0.0] * 100
    data = [id2emb[ix] for ix in range(len(word2ix))]

    return data


if __name__ == '__main__':
    parser_args = init_arg_parser()
    main(parser_args)
