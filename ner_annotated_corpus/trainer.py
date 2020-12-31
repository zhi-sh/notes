# -*- coding: utf-8 -*-
# @DateTime :2020/12/31 下午10:37
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup

from datasets import CustomData


class Trainer:
    def __init__(self, conf):
        self.conf = conf
        self.device = conf.device
        self.tokenizer = BertTokenizer.from_pretrained(conf.bert_model)
        self._dataloader(f"{conf.data_path}/ner_dataset.csv")

    def fit(self, model):
        model.to(self.device)
        for epoch in range(self.conf.epochs):
            self.train_one_epoch(model)
            self.validation(model)

    def train_one_epoch(self, model):
        model.train()
        pass

    def validation(self, model):
        model.eval()
        pass

    def validation(self):

    def _dataloader(self, fpath, seed=2020):
        r'''在训练阶段使用RandomSampler, 在测试阶段使用SequentialSampler'''
        dset = CustomData(fpath, self.tokenizer)
        input_ids, attention_masks, labels = dset.get_data()
        train_inputs, valid_inputs, train_tags, valid_tags = train_test_split(input_ids, labels, random_state=seed, test_size=0.1)
        train_masks, valid_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=seed, test_size=0.1)

        train_data = TensorDataset(torch.tensor(train_inputs), torch.tensor(train_masks), torch.tensor(train_tags))
        train_sampler = RandomSampler(train_data)
        self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.conf.bs)

        valid_data = TensorDataset(torch.tensor(valid_inputs), torch.tensor(valid_masks), torch.tensor(valid_tags))
        valid_sampler = SequentialSampler(valid_data)
        self.valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=self.conf.bs)

    def _get_optimizer(self, model):
        if hasattr(self.conf, 'full_finetuning') and self.conf.full_finetuning:
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01
                },
                {
                    'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0
                }
            ]
        else:
            param_optimizer = list(model.classifier.named_parameters())
            optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.conf.lr, eps=self.conf.eps)

        total_steps = len(self.train_dataloader) * self.conf.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        return optimizer, scheduler
