# -*- coding:utf-8 -*-
import json
import pandas as pd
import jieba
import args
import re

args = args.init_arg_parser()


def DUMA_pre():
    with open(args.data_dir + '/train.json', 'r', encoding='utf-8')as f:  # 读入json文件
        train_data = json.load(f)

    train_df = []
    for i in range(len(train_data)):
        data = train_data[i]
        content = data['Content']
        questions = data['Questions']
        for question in questions:
            choice_concat = ""
            for idx, choice in enumerate(question['Choices']):
                choice_concat += str(choice)
            ans = question['Answer']
            dict_ = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            one_data = {'Content': content, 'Q_id': question['Q_id'],
                        'qa_pair': question['Question'] + str(choice_concat),
                        'Answer': ans, 'label': dict_[ans]}
            train_df.append(one_data)

    train_df = pd.DataFrame(train_df)
    print(train_df.head())
    print(train_df.size)

    with open(args.data_dir + '/validation.json', 'r', encoding='utf-8')as f:
        test_data = json.load(f)

    test_df = []

    for i in range(len(test_data)):
        data = test_data[i]
        content = data['Content']
        questions = data['Questions']
        for question in questions:
            choice_concat = ""
            for idx, choice in enumerate(question['Choices']):
                choice_concat += str(choice)
            one_data = {'Content': content, 'Q_id': question['Q_id'],
                        'qa_pair': question['Question'] + str(choice_concat),
                        'label': 0}
            test_df.append(one_data)

    test_df = pd.DataFrame(test_df)
    print(test_df.head())
    print(test_df.size)
    train_df.to_csv(args.data_dir + 'train_pqa.csv', index=False)
    test_df.to_csv(args.data_dir + 'test_pqa.csv', index=False)


def DUMA_pre_0_1():
    with open(args.data_dir + '/train.json', 'r', encoding='utf-8')as f:  # 读入json文件
        train_data = json.load(f)

    train_df = []
    for i in range(len(train_data)):
        data = train_data[i]
        content = data['Content']
        questions = data['Questions']
        for question in questions:
            for idx, choice in enumerate(question['Choices']):
                ans = question['Answer']
                dict_ = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                one_data = {'Content': content, 'Q_id': question['Q_id'], 'qa_pair': question['Question'] + str(choice),
                            'Answer': ans, 'label': dict_[ans]}
                train_df.append(one_data)

    train_df = pd.DataFrame(train_df)
    print(train_df.head())
    print(train_df.size)

    with open(args.data_dir + '/validation.json', 'r', encoding='utf-8')as f:
        test_data = json.load(f)

    test_df = []

    for i in range(len(test_data)):
        data = test_data[i]
        content = data['Content']
        questions = data['Questions']
        for question in questions:
            for idx, choice in enumerate(question['Choices']):
                one_data = {'Content': content, 'Q_id': question['Q_id'], 'qa_pair': question['Question'] + str(choice),
                            'label': 0}
                test_df.append(one_data)

    test_df = pd.DataFrame(test_df)
    print(test_df.head())
    print(test_df.size)
    train_df.to_csv(args.data_dir + 'train_pqa_0_1.csv', index=False)
    test_df.to_csv(args.data_dir + 'test_pqa_0_1.csv', index=False)


def main():
    with open(args.data_dir + '/train.json', 'r', encoding='utf-8')as f:  # 读入json文件
        train_data = json.load(f)

    train_df = []

    stop_words_list = []
    with open(args.stopwords_path, 'r', encoding='utf-8')as f:  # 读入json文件
        lines = f.readlines()
        for line in lines:
            lline = line.strip()
            stop_words_list.append(lline)

    for i in range(len(train_data)):  # 将每个文章-问题-答案作为一条数据
        data = train_data[i]
        content = data['Content']
        questions = data['Questions']
        content_new = ""
        # cut = jieba.cut(content)
        # for c in cut:
        #     if c not in stop_words_list:
        #         if c != " ":
        #             content_new += c

        for question in questions:
            # punctuation = '[\s+\.\!\/\_,$%^&*\\\(\)\（\）\'\"”“，。：．下列]+'
            # question['Question'] = re.sub(punctuation, "", str(question['Question']))
            # question['Content'] = re.sub(punctuation, "", str(content_new))
            # choi = []
            # for choice in question['Choices']:
            #     choi.append(re.sub(punctuation, "", str(choice)))
            # question['Choices'] = choi
            question['Content'] = content_new
            train_df.append(question)

    train_df = pd.DataFrame(train_df)

    with open('./data/validation.json', 'r', encoding='utf-8')as f:
        test_data = json.load(f)

    test_df = []

    for i in range(len(test_data)):
        data = test_data[i]
        content = data['Content']
        questions = data['Questions']
        cls = data['Type']
        diff = data['Diff']

        content_new = ""
        cut = jieba.cut(content)
        for c in cut:
            if c not in stop_words_list:
                if c != " ":
                    content_new += c

        for question in questions:
            question['Content'] = content_new
            question['Type'] = cls
            question['Diff'] = diff
            test_df.append(question)

    test_df = pd.DataFrame(test_df)
    train_df.to_csv(args.data_dir + 'train_stop.csv', index=False)
    test_df.to_csv(args.data_dir + 'test_stop.csv', index=False)
    print('finish')


if __name__ == '__main__':
    DUMA_pre()
