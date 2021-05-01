# -*- coding:utf-8 -*-
import torch.nn as nn
import torch
from argparse import Namespace
from transformers import BertPreTrainedModel, LongformerModel
from transformers import BertModel, LongformerConfig
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MultipleChoiceModelOutput
from transformers.models.longformer.modeling_longformer import LongformerPreTrainedModel

from bertBaseDistribute.LongFormer.longformer.sliding_chunks import pad_to_window_size
from bertBaseDistribute.LongFormer.longformer.longformer import Longformer
import logging

logger = logging.getLogger("Chn")


class LongformerForMultipleChoice(nn.Module):
    def __init__(self, args, pre_model):
        super().__init__()
        self.args = args
        self.config = LongformerConfig.from_pretrained(pre_model, output_hidden_states=True)

        self.longformer = LongformerModel(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, 1)

        # self.init_weights()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, global_attention_mask=None,
                head_mask=None, labels=None, position_ids=None, inputs_embeds=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]  # 4

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # set global attention on question tokens
        if global_attention_mask is None and input_ids is not None:
            logger.info("Initializing global attention on multiple choice...")
            # put global attention on all tokens after `config.sep_token_id`
            global_attention_mask = torch.stack(
                [
                    # torch.Size([8, 4, 1024])   sep_token_id=2 应该是102呀
                    self._compute_global_attention_mask(input_ids[:, i], 102, before_sep_token=False)
                    for i in range(num_choices)
                ],
                dim=1,
            )

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_global_attention_mask = (
            global_attention_mask.view(-1, global_attention_mask.size(-1))
            if global_attention_mask is not None
            else None
        )
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.longformer(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            global_attention_mask=flat_global_attention_mask,
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

        outputs = (reshaped_logits,)
        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)

    def _get_question_end_index(self, input_ids, sep_token_id):
        """
        Computes the index of the first occurance of `sep_token_id`.
        """
        sep_token_indices = (input_ids == sep_token_id).nonzero()
        batch_size = input_ids.shape[0]

        assert sep_token_indices.shape[1] == 2, "`input_ids` should have two dimensions"
        assert (
                sep_token_indices.shape[0] == 3 * batch_size
        ), f"There should be exactly three separator tokens: {sep_token_id} in every sample for questions answering. You might also consider to set `global_attention_mask` manually in the forward function to avoid this error."
        return sep_token_indices.view(batch_size, 3, 2)[:, 0, 1]

    def _compute_global_attention_mask(self, input_ids, sep_token_id, before_sep_token=True):
        """
        Computes global attention mask by putting attention on all tokens before `sep_token_id` if `before_sep_token is
        True` else after `sep_token_id`.
        """
        question_end_index = self._get_question_end_index(input_ids, sep_token_id)
        question_end_index = question_end_index.unsqueeze(dim=1)  # size: batch_size x 1
        # bool attention mask with True in locations of global attention
        attention_mask = torch.arange(input_ids.shape[1], device=input_ids.device)
        if before_sep_token is True:
            attention_mask = (attention_mask.expand_as(input_ids) < question_end_index).to(torch.uint8)
        else:
            # last token is separation token and should not be counted and in the middle are two separation tokens
            attention_mask = (attention_mask.expand_as(input_ids) > (question_end_index + 1)).to(torch.uint8) * (
                    attention_mask.expand_as(input_ids) < input_ids.shape[-1]
            ).to(torch.uint8)

        return attention_mask


class LongformerClassifier(nn.Module):

    def __init__(self, init_args):
        super().__init__()
        if isinstance(init_args, dict):
            # for loading the checkpoint, pl passes a dict (hparams are saved as dict)
            init_args = Namespace(**init_args)
        config_path = init_args.config_path or init_args.model_dir
        checkpoint_path = init_args.checkpoint_path or init_args.model_dir
        logger.info(f'loading model from config: {config_path}, checkpoint: {checkpoint_path}')
        config = LongformerConfig.from_pretrained(config_path)
        config.attention_mode = init_args.attention_mode
        logger.info(f'attention mode set to {config.attention_mode}')
        self.model_config = config
        self.model = Longformer.from_pretrained(checkpoint_path, config=config)
        self.tokenizer.model_max_length = self.model.config.max_position_embeddings
        self.hparams = init_args
        self.hparams.seqlen = self.model.config.max_position_embeddings
        self.classifier = nn.Linear(config.hidden_size, init_args.num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        input_ids, attention_mask = pad_to_window_size(
            input_ids, attention_mask, self.model_config.attention_window[0], self.tokenizer.pad_token_id)
        attention_mask[:, 0] = 2  # global attention for the first token
        # use Bert inner Pooler
        output = self.model(input_ids, attention_mask=attention_mask)[1]
        # pool the entire sequence into one vector (CLS token)
        # output = output[:, 0, :]
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            loss = loss_fct(logits.view(-1, self.hparams.num_labels), labels.view(-1))

        return logits, loss
