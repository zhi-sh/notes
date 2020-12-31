# -*- coding: utf-8 -*-
# @DateTime :2020/12/31 下午9:26
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
from transformers import BertForTokenClassification


class Model(nn.Module):
    def __init__(self, conf):
        super(Model, self).__init__()
        self.conf = conf
        self.bert = BertForTokenClassification.from_pretrained(conf.bert_model, conf.num_labels, output_attentions=False, output_hidden_states=False)

    def forward(self, batch):
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask', None)
        token_type_ids = batch.get('token_type_ids', None)
        labels = batch.get('labels', None)
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        return outputs
