# -*- coding:utf-8 -*-
import json
import pandas as pd
import jieba
import args
import os
import jsonlines
import random
import time
import re
from TranslateAPI import Trans

args = args.init_arg_parser()


def data_longformer(arg_in, mode="train"):
    def replace_placeholder(str, opt):
        list = str.split(' ')
        for i in range(len(list)):
            if list[i] == '（）':
                list[i] = opt
        final_opt = " ".join(list)
        return final_opt

    data_list = []
    labels = []
    file_path = arg_in.data_dir + "train_baseline.json" if mode == 'train' else arg_in.data_dir + "test_baseline.json"
    with open(file_path, mode='r') as json_file:
        reader = jsonlines.Reader(json_file)
        for instance in reader:
            article = instance['article'].strip().replace(' ', '').replace('\u3000', '') \
                .replace('\n', '').replace('\xa0', '')
            question = instance['question'].replace('\u3000', '') \
                .replace('\n', '').replace('\xa0', '')
            opt1 = instance['option_0']
            opt1 = replace_placeholder(question, opt1)
            opt2 = instance['option_1']
            opt2 = replace_placeholder(question, opt2)
            opt3 = instance['option_2']
            opt3 = replace_placeholder(question, opt3)
            opt4 = instance['option_3']
            opt4 = replace_placeholder(question, opt4)

            data_list.append({
                'text': article + '[SEP]' + question + '[SEP]' + opt1 + '[SEP]' + opt2 + '[SEP]' + opt3 + '[SEP]' + opt4,
                'label': instance['label'] if mode == "train" else -1,
            })

    train_df = pd.DataFrame(data_list)
    print(train_df.head())
    print(train_df.size)
    train_df.to_csv(args.data_dir + mode + '_baseline.csv', index=False)


def data_enhancement_trans(arg_in):
    train_df_trans = pd.read_csv(args.data_dir + "train_label_trans.csv")
    train_df_label = pd.read_csv(args.data_dir + "train_label.csv")
    train_df_label = train_df_label.iloc[15263:]

    # # 判断断掉时正在处理的Q_id
    # tail_data_trans = train_df_trans.iloc[-1, :]
    # tail_data_label = train_df_label.iloc[-1, :]
    # if tail_data_trans['Q_id'] == tail_data_label['Q_id']:
    #     print("任务已完成")
    #     return
    # else:  # 否则切分数据，切到最后停止时处理的数据
    #     print("从")
    #     last_index = train_df_label[train_df_label['Q_id'] == tail_data_trans['Q_id']].index.values[0]
    #     last_index_label = train_df_label[train_df_label['Q_id'] == tail_data_trans['Q_id']].index.values[0]
    #     train_df = train_df_label.iloc[last_index:]

    # train_df = train_df.drop(['Unnamed: 0', 'num_choices', 'content_len'], axis=1)
    column = train_df_label.columns
    train_df_allup = []
    trans = Trans()
    count = 0

    last_content = ""
    last_content_zh = ""
    last_question = ""
    last_question_zh = ""
    for i, row in train_df_label.iterrows():
        row.Content = row.Content.strip().replace(' ', '').replace('\n', '')
        content = row.Content
        question = row.Question
        choices = row.Choices

        one_data = {
            'Content': content.strip().replace(' ', '').replace('\u3000', '').replace('\n', '').replace('\xa0', ''),
            'Q_id': row['Q_id'],
            'Question': row['Question'].replace('\n', ''),
            'Choices': row['Choices'],
            'Answer': row['Answer'], 'label': row['label']}
        # train_df_allup.append(one_data)
        train_df_trans.loc[train_df_trans.shape[0]] = one_data

        if content != last_content:
            en_content = trans.translate(content, from_lang='zh', to_lang='en')
            if check_error(en_content):
                print(en_content)
                print(1)
                break
            zh_content = trans.translate(en_content, from_lang='en', to_lang='zh')
            last_content_zh = zh_content
            if check_error(zh_content):
                print(zh_content)
                print(2)
                break
        else:
            zh_content = last_content_zh

        if last_question != question:
            en_question = trans.translate(question, from_lang='zh', to_lang='en')
            if check_error(en_question):
                print(en_question)
                print(3)
                break

            zh_question = trans.translate(en_question, from_lang='en', to_lang='zh')
            last_question_zh = zh_question
            if check_error(zh_question):
                print(zh_question)
                print(4)
                break
        else:
            zh_question = last_question_zh

        list_choices = choices[2:-2].split('\', \'')
        new_choices = []
        flag = 0
        for choice in list_choices:
            en_choice = trans.translate(choice, from_lang='zh', to_lang='en')
            if check_error(en_choice):
                print(en_choice)
                flag = 5
                break
            zh_choice = trans.translate(en_choice, from_lang='en', to_lang='zh')
            if check_error(zh_choice):
                print(zh_choice)
                flag = 6
                break
            new_choices.append(zh_choice)

        if flag == 5 or flag == 6:
            print(flag)
            break

        last_content = content
        last_question = question

        count += 1
        one_data = {
            'Content': zh_content.strip().replace(' ', '').replace('\u3000', '').replace('\n', '').replace('\xa0', ''),
            'Q_id': row['Q_id'],
            'Question': zh_question.replace('\n', ''),
            'Choices': str(new_choices),
            'Answer': row['Answer'], 'label': row['label']}
        print("count:", count, one_data)
        # train_df_allup.append(one_data)
        train_df_trans.loc[train_df_trans.shape[0]] = one_data
        # df = pd.DataFrame(train_df_allup)
        train_df_trans.to_csv(args.data_dir + 'train_label_trans.csv', index=False)

    print("-------------------------------", count)
    # df = pd.DataFrame(train_df_allup)
    # df.to_csv(args.data_dir + 'train_label_trans.csv', index=False)
    train_df_trans.to_csv(args.data_dir + 'train_label_trans.csv', index=False)


def check_error(str):
    return "error!!!!:" in str


def data_enhancement_sentence_order(arg_in):
    train_df = pd.read_csv(arg_in.train_path)
    train_df = train_df.drop(['Unnamed: 0', 'num_choices', 'content_len'], axis=1)
    column = train_df.columns
    count = 0
    train_df_allup = []
    for i, row in train_df.iterrows():
        row.Content = row.Content.strip().replace(' ', '').replace('\n', '')
        content = row.Content
        question_len = len(row.Question)
        choices = row.Choices

        flag = 0
        for choice in choices:
            if question_len + len(choice) > 100:
                flag = 1
                break
        if flag == 1:
            continue

        count += 1
        pattern = r'\.|/|;|。|；|！|？'
        result_list = re.split(pattern, content)
        shuffle_n(result_list, len(result_list))
        new_content = '。'.join(result_list)

        one_data = {
            'Content': new_content.strip().replace(' ', '').replace('\u3000', '').replace('\n', '').replace('\xa0', ''),
            'Q_id': row['Q_id'],
            'Question': row['Question'].replace('\n', ''),
            'Choices': row['Choices'],
            'Answer': row['Answer'], 'label': row['label']}
        print(one_data)
        train_df.loc[train_df.shape[0]] = one_data

    print("-------------------------------", count)
    train_df.to_csv(args.data_dir + 'train_label_order_allup.csv', index=False)


def data_enhancement_sentence_order_bingpai(arg_in):
    train_df = pd.read_csv(arg_in.train_path)
    train_df = train_df.drop(['Unnamed: 0', 'num_choices', 'content_len'], axis=1)
    column = train_df.columns
    count = 0
    train_df_allup = []
    for i, row in train_df.iterrows():
        row.Content = row.Content.strip().replace(' ', '').replace('\n', '')
        content = row.Content
        question_len = len(row.Question)
        choices = row.Choices

        one_data = {
            'Content': content.strip().replace(' ', '').replace('\u3000', '').replace('\n', '').replace('\xa0', ''),
            'Q_id': row['Q_id'],
            'Question': row['Question'].replace('\n', ''),
            'Choices': row['Choices'],
            'Answer': row['Answer'], 'label': row['label']}
        train_df_allup.append(one_data)

        flag = 0
        list_choices = choices[2:-2].split('\', \'')
        for choice in list_choices:
            if question_len + len(choice) > 100:
                flag = 1
                break
        if flag == 1:
            continue

        count += 1
        pattern = r'\.|/|;|。|；|！|？'
        result_list = re.split(pattern, content)
        shuffle_n(result_list, len(result_list))
        new_content = '。'.join(rs for rs in result_list if re != ' ')

        one_data = {
            'Content': new_content.strip().replace(' ', '').replace('\u3000', '').replace('\n', '').replace('\xa0', ''),
            'Q_id': row['Q_id'],
            'Question': row['Question'].replace('\n', ''),
            'Choices': row['Choices'],
            'Answer': row['Answer'], 'label': row['label']}
        print(one_data)
        train_df_allup.append(one_data)

    print("-------------------------------", count)
    df = pd.DataFrame(train_df_allup)
    df.to_csv(args.data_dir + 'train_label_order_bingpai.csv', index=False)


def shuffle_n(arr, n):
    random.seed(time.time())
    for i in range(len(arr) - 1, len(arr) - n - 1, -1):
        # print(i)
        j = random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]


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
    # DUMA_pre()
    # data_enhancement_sentence_order_bingpai(args)
    # data_enhancement_trans(args)
    data_longformer(args)
