# -*- coding:utf-8 -*-
import argparse


def init_arg_parser():
    arg_parser = argparse.ArgumentParser(description='Haihua RC')
    # 多卡配置
    arg_parser.add_argument('--mult_gpu', default=True)  # 使用多卡还是单卡
    arg_parser.add_argument('--sigle_device', default=0)  # 单卡的时候使用哪个卡
    arg_parser.add_argument('--nprocs', default=4, type=int)  # 使用几个卡
    arg_parser.add_argument('--tcp', default='tcp://127.0.0.1:23456')  # 使用几个卡
    arg_parser.add_argument('--backend', default='nccl')  # 使用几个卡

    # 数据预处理时的参数
    arg_parser.add_argument('--train_size', default=0.8)  # 训练集所占比例
    arg_parser.add_argument('--max_len', default=400)  # bert输入句子的最大长度
    arg_parser.add_argument('--use_label_smoothing', default=True)  # 是否使用软标签
    arg_parser.add_argument('--label_smooth', default=0.1)  # 软标签阈值
    arg_parser.add_argument('--add_key_words', default=False)  # 是否添加关键词
    arg_parser.add_argument('--key_words_num', default=2)  # 添加关键词的数目
    arg_parser.add_argument('--max_df', default=1.0)  # tfidf中的词在句子中出现的次数占据所有句子的最大比
    arg_parser.add_argument('--min_df', default=1)  # tfidf中的词在句子中出现的最少次数
    arg_parser.add_argument('--rm_stopwords', default=True)  # tfidf中的词在句子中出现的最少次数
    arg_parser.add_argument('--label_info', default={
        '100': 0, '101': 1, '102': 2, '103': 3, '104': 4, '106': 5, '107': 6, '108': 7, '109': 8, '110': 9, '112': 10,
        '113': 11, '114': 12, '115': 13, '116': 14})  # tfidf中的词在句子中出现的最少次数

    # 训练时参数
    arg_parser.add_argument('--weight_decay', default=0.0, type=float)  # 权重衰减
    arg_parser.add_argument('--warmup_steps', default=0, type=float)  # warmup_steps
    arg_parser.add_argument('--use_early_stop', default=False)  # 是否早停
    arg_parser.add_argument('--early_stop_num', default=1e-5, type=float)  # 耐心
    arg_parser.add_argument('--accum_iter', default=8, type=int)  # 梯度累加数量
    arg_parser.add_argument('--num_workers', default=4, type=int)  # pytorch dataloader num_worker 数量
    arg_parser.add_argument('--stop_num', default=4)  # 当评价指标在stop_num内不在增长时，训练停止
    arg_parser.add_argument('--trail_train', default=True)  # 当评价指标在stop_num内不在增长时，训练停止
    arg_parser.add_argument('--seed', default=15484)  # 随机种子
    arg_parser.add_argument('--epochs', default=10)  # 数据训练多少轮
    # arg_parser.add_argument('--batch_size', default=2)  # 模型一次输入的数据
    arg_parser.add_argument('--train_bs', default=4)  # 训练集batch
    arg_parser.add_argument('--valid_bs', default=4)  # 验证集batch
    arg_parser.add_argument('--save_model', default=False)  # 是否保存模型
    arg_parser.add_argument('--use_amsgrad', default=False)  # 是否使用amsgrad
    arg_parser.add_argument('--is_train', default=True)  # 是否是训练
    arg_parser.add_argument('--use_kfold', default=True)  # 是否使用交叉验证
    arg_parser.add_argument('--kfold_num', default=5)  # 采用几折交叉验证
    arg_parser.add_argument('--use_fgm', default=True)  # 是否使用fgm对抗训练
    arg_parser.add_argument('--use_pgd', default=False)  # 是否使用pgd对抗训练
    arg_parser.add_argument('--use_FreeLB', default=False)  # 是否使用FreeLB对抗训练
    arg_parser.add_argument('--k_pdg', default=3)  # pdg对抗训练的次数
    arg_parser.add_argument('--loss_step', default=2)  # 模型几次loss后更新参数
    arg_parser.add_argument('--out_dev_result', default=False)  # 是否输出评估集每个标签的得分情况

    # 各种文件路径
    arg_parser.add_argument('--bert_chinese_wwm_ext', default="/home/zuoyuhui/DataGame/haihuai_RC/chinese-bert-wwm-ext")
    arg_parser.add_argument('--hfl_chinese_roberta_wwm_ext', default='/data2/roberta/hfl_chinese_roberta_wwm_ext')
    arg_parser.add_argument('--RoBERTa_zh_L12', default='/data2/roberta/RoBERTa_zh_L12')
    arg_parser.add_argument('--robert_chinese_pytorch', default='/data2/roberta/robert-chinese-pytorch')
    arg_parser.add_argument('--longformer_base', default='/data2/pre-model/longformer')
    arg_parser.add_argument('--hfl_chinese_roberta_wwm_ext_large',
                            default='/data2/roberta/hfl_chinese_roberta_wwm_ext_large')
    arg_parser.add_argument('--data_dir', metavar='DIR', default="/home/zuoyuhui/DataGame/haihuai_RC/data/")
    arg_parser.add_argument('--train_path',
                            default="/home/zuoyuhui/DataGame/haihuai_RC/data/train_label.csv")  # 训练集的文件路径
    arg_parser.add_argument('--test_path', default='/home/zuoyuhui/DataGame/haihuai_RC/data/test.csv')  # 测试集的文件路径
    arg_parser.add_argument('--output_path', default='/data2/code/bertBaseDistribute/models')  # 模型输出路径
    arg_parser.add_argument('--stopwords_path', default='/data2/stopword_set/hit_stopwords1.txt')  # 停用词路径
    arg_parser.add_argument('--save_model_name', default='bert_damu_ema_early_{}_fold_{}.pt')  # 模型保存名字

    # 模型内各种参数
    arg_parser.add_argument('--lr', '--learning-rate', default=2e-5)  # 学习率
    arg_parser.add_argument('--fc_lr', default=2e-3)  # 分类层学习率
    arg_parser.add_argument('--other_lr', default=2e-3)  # 其他层学习率
    arg_parser.add_argument('--bert_lr', default=2e-5)  # bert层学习率
    arg_parser.add_argument('--fc_dropout', default=0.5)  # 分类层dropout
    arg_parser.add_argument('--use_bert_dropout', default=False)  # bert输出层dropout
    arg_parser.add_argument('--bert_dropout', default=0.15)  # bert输出层dropout
    arg_parser.add_argument('--bert_dim', default=1024)  # bert的输出向量维度
    arg_parser.add_argument('--lstm_hidden_dim', default=512)  # lstm的隐藏层
    arg_parser.add_argument('--bilstm', default=True)  # lstm是否双向
    arg_parser.add_argument('--output_hidden_states', default=True)  # 是否输出隐藏层
    arg_parser.add_argument('--use_cls', default=False)  # 是否使用cls
    arg_parser.add_argument('--class_num', default=4)  # 输出的维度

    # DUMA
    arg_parser.add_argument('--right_len', default=200)  # 问答对的最大总长度
    arg_parser.add_argument('--left_len', default=200)  # 文章最大长度
    arg_parser.add_argument('--coattn_head', default=8)  # coatt 多头数量
    arg_parser.add_argument('--coattn_size', default=64)  # coatt 每个头的维度
    arg_parser.add_argument('--duma_layer', default=2)  # coatt 层数
    arg_parser.add_argument('--coattn_drop', default=0.1)  # coattn dropout

    # bert model
    # {
    #     "architectures": [
    #         "BertForMaskedLM"
    #     ],
    #     "attention_probs_dropout_prob": 0.1,
    #     "directionality": "bidi",
    #     "hidden_act": "gelu",
    #     "hidden_dropout_prob": 0.1,
    #     "hidden_size": 768,
    #     "initializer_range": 0.02,
    #     "intermediate_size": 3072,
    #     "layer_norm_eps": 1e-12,
    #     "max_position_embeddings": 512,
    #     "model_type": "bert",
    #     "num_attention_heads": 12,
    #     "num_hidden_layers": 12,
    #     "output_past": true,
    #     "pad_token_id": 0,
    #     "pooler_fc_size": 768,
    #     "pooler_num_attention_heads": 12,
    #     "pooler_num_fc_layers": 3,
    #     "pooler_size_per_head": 128,
    #     "pooler_type": "first_token_transform",
    #     "type_vocab_size": 2,
    #     "vocab_size": 21128
    # }

    args = arg_parser.parse_args()
    return args
