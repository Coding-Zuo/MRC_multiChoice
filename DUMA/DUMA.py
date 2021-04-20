# -*- coding:utf-8 -*-
from args import init_arg_parser
import torch.nn as nn
import numpy as np
import torch
import json
import torch.nn.functional as F
import copy
from transformers import BertPreTrainedModel, RobertaModel
from transformers import BertModel, BertConfig
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MultipleChoiceModelOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel


def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)

        if total_length <= max_length:
            break

        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()

        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()


class InputExample_MCinQA(object):
    """A single training/dev/test example for Multiple Choice RC without paragraph (only question and candidate) task"""

    def __init__(self, guid, question, choices=[], label=None, q_id=None):
        self.guid = guid
        self.question = question
        self.choices = choices  # list in order
        self.label = label
        self.q_id = q_id

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class CoAttention(nn.Module):
    def __init__(self, args):
        super(CoAttention, self).__init__()
        self.model_dim = 768  # 模型维度
        self.coattn_size = args.coattn_size  # qkv 维度 = 64
        self.head = args.coattn_head  # 多头头数
        self.dim = args.coattn_head * args.coattn_size
        self.W_Q = nn.Linear(self.model_dim, self.dim)  # in 768 out 512=64*8
        self.W_K = nn.Linear(self.model_dim, self.dim)
        self.W_V = nn.Linear(self.model_dim, self.dim)

    def forward(self, Q, K, V, attn_mask):
        # q/k/v :[batch_size, seq_len=200, d_model=768]
        residual, batch_size, seq_len = Q, Q.size(0), Q.size(1)  # [batch_size, seq_len=200, d_model=768]
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B,H,S,W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.head, self.coattn_size).transpose(1,
                                                                                      2)  # q_s:[batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.head, self.coattn_size).transpose(1,
                                                                                      2)  # k_s:[batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.head, self.coattn_size).transpose(1,
                                                                                      2)  # v_s:[batch_size, n_heads, seq_len, d_k]
        attn_mask = attn_mask.data.eq(0).unsqueeze(1).expand(batch_size, seq_len, seq_len)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.head, 1,
                                                  1)  # attn_mask:[batch_size, n_heads, seq_len, seq_len]
        # context :[batch_size, n_heads, seq_len, d_v]
        # attn:[batch_size, n_heads, seq_len, seq_len]
        context = self._scaledDotProductAttention(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            self.head * self.coattn_size)  # [batch_size, seq_len, d_model]
        output = nn.Linear(self.head * self.coattn_size, self.model_dim).cuda()(context)
        output = nn.Dropout(0.1)(output)
        return nn.LayerNorm(self.model_dim).cuda()(output + residual)  # output:[batch_size, seq_len, d_model]

    def _scaledDotProductAttention(self, q_s, k_s, v_s, attn_mask):
        # torch.Size([8, 8, 200, 64])
        k_s = k_s.transpose(-1, -2)
        # torch.Size([8, 8, 64, 200])
        scores = torch.matmul(q_s, k_s / np.sqrt(self.coattn_size))  # scores: [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, 1e-9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, v_s)
        return context


class DUMA(nn.Module):
    def __init__(self, args, pre_model):
        super().__init__()
        self.left = args.left_len
        self.right = args.right_len
        self.coattn_head = args.coattn_head
        self.duma_layer = args.duma_layer
        self.coattn_drop = args.coattn_drop
        self.size = [-1, self.left + self.right]

        self.poola = nn.AdaptiveAvgPool1d(1)
        self.poolb = nn.AdaptiveAvgPool1d(1)

        self.MHA1 = CoAttention(args)
        self.MHA2 = CoAttention(args)
        self.args = args

        self.bert_config = BertConfig.from_pretrained(pre_model, output_hidden_states=True)
        self.bert = BertModel(self.bert_config).from_pretrained(pre_model)
        self.dropout = nn.Dropout(self.bert_config.hidden_dropout_prob)
        self.co_dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(768 * 2, 1)
        self.classifier = nn.Linear(self.bert_config.hidden_size, 1)
        # self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 将bert三个输入展平 输入到bertmodel
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # 隐层输出
        # output_layer = self.bert.get_sequence_output()  # 这个获取每个token的output 输出[batch_size, seq_length, embedding_size] 如果做seq2seq 或者ner 用这个
        # pooled_output = self.bert.get_pooled_output()  # 这个获取句子的output
        """
            last_hidden_state: [32=4*batch, seq_len,768]
            pooler_ouput: [32=4*batch,768]
        """
        last_hidden_state = outputs[0]
        pooled_output = outputs[1]  # CLS https://www.cnblogs.com/webbery/p/12167552.html
        # last_hidden_state = last_hidden_state.view(-1, 4, self.args.max_len, 768)
        # [4batch, seqlen, 768] -> [4batch,seqlen/2,768]
        passage_hidden, qa_hidden = torch.split(last_hidden_state, split_size_or_sections=[self.right, self.left],
                                                dim=1)
        # [4batch, seqlen] -> [4batch,seqlen/2]
        passage_attention_mask, qa_attention_mask = torch.split(attention_mask,
                                                                split_size_or_sections=[self.right, self.left],
                                                                dim=1)

        mh1 = self.MHA1(passage_hidden, qa_hidden, qa_hidden, qa_attention_mask)  # torch.Size([8, 200, 768])
        mh2 = self.MHA2(qa_hidden, passage_hidden, passage_hidden, passage_attention_mask)  # torch.Size([8, 200, 768])
        mh1 = self.MHA1(mh1, mh2, mh2, qa_attention_mask)  # torch.Size([8, 200, 768])
        mh2 = self.MHA2(mh2, mh1, mh1, passage_attention_mask)  # torch.Size([8, 200, 768])

        # mh1 = mh1.view(-1, 4, qa_hidden.shape[1], qa_hidden.shape[2])
        # mh2 = mh2.view(-1, 4, qa_hidden.shape[1], qa_hidden.shape[2])
        mh1 = F.adaptive_avg_pool2d(mh1, (1, last_hidden_state.shape[2])).squeeze(dim=1)
        mh2 = F.adaptive_avg_pool2d(mh2, (1, last_hidden_state.shape[2])).squeeze(dim=1)
        # b = torch.nn.functional.adaptive_avg_pool2d(a, (1, 1))  # 自适应池化，指定池化输出尺寸为 1 * 1

        mh1 = torch.cat((mh1, mh2), 1)  # torch.Size([8, 1536])
        mh1 = self.co_dropout(mh1)  # torch.Size([8, 1536])
        mh1 = self.dense(mh1)  # torch.Size([8, 1])
        mh1 = nn.Softmax(dim=1)(mh1.view(-1, 4))

        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        # reshaped_logits = logits.view(-1, num_choices)

        outputs = (mh1,)  # add hidden states and attention if they are here

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


class BertForMultipleChoiceBiLSTM(nn.Module):
    def __init__(self, args, pre_model):
        super().__init__()
        self.args = args
        self.bert_config = BertConfig.from_pretrained(pre_model, output_hidden_states=True)
        self.bert = BertModel(self.bert_config).from_pretrained(pre_model)

        self.lstm = nn.LSTM(self.bert_config.hidden_size, args.lstm_hidden_size,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(args.lstm_hidden_size * 2, args.lstm_hidden_size,
                          num_layers=1, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(self.bert_config.hidden_dropout_prob)
        self.classifier = nn.Linear(args.lstm_hidden_size * 2 * 3 + self.bert_config.hidden_size, 1)
        # self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 将bert三个输入展平 输入到bertmodel
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
        # 隐层输出
        pooled_output = outputs[1]  # torch.Size([16, 768]) CLS https://www.cnblogs.com/webbery/p/12167552.html
        bert_output = outputs[0]  # torch.Size([16, 400, 768])
        h_lstm, _ = self.lstm(bert_output)  # [batch_size, seq, output*2] torch.Size([16, 400, 1024])
        h_gru, hh_gru = self.gru(h_lstm)  # torch.Size([16, 400, 1024])  torch.Size([2, 16, 512])
        hh_gru = hh_gru.view(-1, 2 * self.args.lstm_hidden_size)  # torch.Size([16, 1024])

        avg_pool = torch.mean(h_gru, 1)  # torch.Size([16, 1024])
        max_pool = torch.max(h_gru, 1)  # torch.Size([16, 1024])

        # print(h_gru.shape, avg_pool.shape, hh_gru.shape, max_pool.shape, pooled_output.shape)
        h_conc_a = torch.cat((avg_pool, hh_gru, max_pool.values, pooled_output),
                             1)  # torch.Size([16, 3840]) 1024*2 + 1024 + 768
        # print(h_conc_a.shape)

        output = self.dropout(h_conc_a)
        logits = self.classifier(output)  # torch.Size([16, 1])
        outputs = nn.functional.softmax(logits, -1)
        outputs = outputs.view(-1, 4)
        outputs = (outputs,)
        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


if __name__ == '__main__':
    pass
