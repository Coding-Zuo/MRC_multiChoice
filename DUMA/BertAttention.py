# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers import BertPreTrainedModel, BertModel, BertConfig


class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state: torch.Tensor, mask: torch.Tensor):
        q = self.fc(hidden_state).squeeze(dim=1)
        mask = mask.view(q.shape[0], -1).unsqueeze(dim=2)
        q = q.masked_fill(mask, -np.inf)  # torch.Size([32, 256, 1]) mask torch.Size([8, 4, 256])
        # w = F.softmax(q, dim=-1).unsqueeze(dim=1)
        w = F.softmax(q, dim=-1).transpose(dim0=1, dim1=2)
        # h = w @ hidden_state  # torch.Size([32, 1, 256, 1])  torch.Size([32, 256, 768])
        h = w.mm(hidden_state)
        return h.squeeze(dim=1)


class AttentionClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super(AttentionClassifier, self).__init__()
        self.attn = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        h = self.attn(hidden_states, mask)
        out = self.fc(h)
        return out


class MultiDropout(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super(MultiDropout, self).__init__()
        self.fc = nn.Linear(2 * hidden_size, num_classes)
        self.dropout = nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)])

    def forward(self, hidden_states: torch.Tensor):
        max_pool, _ = hidden_states.max(dim=1)
        avg_pool = hidden_states.mean(dim=1)
        pool = torch.cat([max_pool, avg_pool], dim=1)
        logits = []
        for dropout in self.dropout:
            out = dropout(pool)
            out = self.fc(out)
            logits.append(out)
        logits = torch.stack(logits, dim=2).mean(dim=2)
        return logits


class BertModelChoice(nn.Module):
    def __init__(self, args, pre_model):
        super(BertModelChoice, self).__init__()
        self.bert_config = BertConfig.from_pretrained(pre_model, output_hidden_states=True)
        self.bert = BertModel(self.bert_config).from_pretrained(pre_model)
        self.dropout = nn.Dropout(self.bert_config.hidden_dropout_prob)
        self.task_classifiers = AttentionClassifier(self.bert_config.hidden_size, 4)

    def forward(self, input_ids, token_type_ids, attention_mask, position_ids=None, head_mask=None, inputs_embeds=None):
        mask = input_ids == 0

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 将bert三个输入展平 输入到bertmodel
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)

        # outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = self.dropout(outputs[0])
        logits = self.task_classifiers(hidden_states, mask)
        outputs = (logits,)
        return outputs
