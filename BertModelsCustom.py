# -*- coding:utf-8 -*-
import torch.nn as nn
import torch
from transformers import BertPreTrainedModel, RobertaModel
from transformers import BertModel
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MultipleChoiceModelOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel


class BertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 将bert三个输入展平 输入到bertmodel
        """
        last_hidden_state: [32=4*batch, seq_len,768]
        pooler_ouput: [32=4*batch,768]
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # 隐层输出
        pooled_output = outputs[1]  # CLS https://www.cnblogs.com/webbery/p/12167552.html

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


class BertForMultipleChoice3Linear(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier3 = nn.Linear(config.hidden_size, 512)
        self.classifier2 = nn.Linear(512, 256)
        self.classifier1 = nn.Linear(256, 1)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 将bert三个输入展平 输入到bertmodel
        """
        last_hidden_state: [32=4*batch, seq_len,768]
        pooler_ouput: [32=4*batch,768]
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # 隐层输出
        pooled_output = outputs[1]  # CLS https://www.cnblogs.com/webbery/p/12167552.html

        # pooled_output = self.dropout(pooled_output)
        logits = self.classifier3(pooled_output)
        logits = self.classifier2(logits)
        logits = self.classifier1(logits)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


class BertForMultipleChoiceBiLSTM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)

        self.lstm = nn.LSTM(config.hidden_size, config.lstm_hidden_size,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(config.lstm_hidden_size * 2, config.lstm_hidden_size,
                          num_layers=1, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 4, 1)
        self.init_weights()

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
        pooled_output = outputs[1]  # CLS https://www.cnblogs.com/webbery/p/12167552.html
        bert_output = outputs[0]
        h_lstm, _ = self.lstm(bert_output)  # [batch_size, seq, output*2]
        h_gru, hh_gru = self.gru(h_lstm)
        hh_gru = hh_gru.view(-1, 2 * self.config.lstm_hidden_size)

        avg_pool = torch.mean(h_gru, 1)
        max_pool = torch.max(h_gru, 1)

        # print(h_gru.shape, avg_pool.shape, hh_gru.shape, max_pool.shape, pooled_output.shape)
        h_conc_a = torch.cat((avg_pool, hh_gru, max_pool, pooled_output), 1)
        # print(h_conc_a.shape)

        output = self.dropout(h_conc_a)
        logits = self.classifier(output)
        outputs = nn.functional.softmax(logits, -1)

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


class RobertaForMultipleChoice(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            labels=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
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

        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        out = MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return out

# class bert_classifi(nn.Module):
#     def __init__(self, args):
#         super(bert_classifi, self).__init__()
#
#         self.output_hidden_states = args.output_hidden_states
#         self.use_bert_dropout = args.use_bert_dropout
#
#         self.bert_model = BertModel.from_pretrained(args.bert_path)
#         for param in self.bert_model.parameters():
#             param.requires_grad = True
#
#         self.bert_dropout = nn.Dropout(args.bert_dropout)
#         self.classifier = nn.Linear(768, 1)
#
#     def forward(self, input_ids, token_type_ids, attention_mask):
#         num_choices = input_ids.shape[1]
#         word_vec, sen_vec, hidden_states = self.bert_model(
#             input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
#             output_hidden_states=self.output_hidden_states
#         )
#         if self.use_bert_dropout:
#             word_vec = self.bert_dropout(word_vec)
#
#         logits = self.classifier(word_vec)
#         reshaped_logits = logits.view(-1, num_choices)
#         outputs = (reshaped_logits,) + outputs[2:]
#
#         return outputs
