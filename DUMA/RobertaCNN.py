# -*- coding:utf-8 -*-
import torch.nn as nn
import numpy as np
import torch
import json
import torch.nn.functional as F
import copy
from transformers import BertPreTrainedModel, RobertaModel
from transformers import BertModel, BertConfig, RobertaConfig
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MultipleChoiceModelOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel


class textCNN(nn.Module):
    def __init__(self, args):
        super(textCNN, self).__init__()
        self.args = args

        Vocab = args.embed_num  ## 已知词的数量
        Dim = args.embed_dim  ##每个词向量长度
        Cla = args.class_num  ##类别数
        Ci = 1  ##输入的channel数
        Knum = args.kernel_num  ## 每种卷积核的数量
        Ks = args.kernel_sizes  ## 卷积核list，形如[2,3,4]

        self.embed = nn.Embedding(Vocab, Dim)  ## 词向量，这里直接随机

        self.convs = nn.ModuleList([nn.Conv2d(Ci, Knum, (K, Dim)) for K in Ks])  ## 卷积层
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(Ks) * Knum, Cla)  ##全连接层

    def forward(self, x):
        x = self.embed(x)  # (N,W,D)

        x = x.unsqueeze(1)  # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # len(Ks)*(N,Knum,W)
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]  # len(Ks)*(N,Knum)

        x = torch.cat(x, 1)  # (N,Knum*len(Ks))

        x = self.dropout(x)
        logit = self.fc(x)
        return logit


class RobertaForMultipleChoice(nn.Module):

    def __init__(self, args, pre_model):
        super().__init__()
        self.config = RobertaConfig.from_pretrained(pre_model, output_hidden_states=True)
        self.roberta = RobertaModel(self.config).from_pretrained(pre_model)
        # self.textCNN = textCNN(args)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, 1)
        # self.init_weights()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None, position_ids=None,
                head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.roberta(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids,
                               attention_mask=flat_attention_mask, head_mask=head_mask,
                               inputs_embeds=flat_inputs_embeds, output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states, return_dict=return_dict, )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,)
        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)
