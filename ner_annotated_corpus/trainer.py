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
        optimizer, scheduler = self._get_optimizer(model)
        for epoch in range(self.conf.epochs):
            train_loss = self.train_one_epoch(model, optimizer, scheduler)
            valid_loss, valid_acc = self.validation(model)

    def train_one_epoch(self, model, optimizer=None, scheduler=None):
        model.train()

        if optimizer is None:
            optimizer, scheduler = self._get_optimizer(model)

        total_loss = 0
        for step, batch in enumerate(self.train_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            batch = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2]
            }
            model.zero_grad()
            outputs = model(batch)
            loss = outputs[0]
            total_loss += loss.item()
            if hasattr(self.conf, 'max_grad_norm') and self.conf.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.conf.max_grad_norm)
            optimizer.step()
            scheduler.step()

        train_loss = total_loss / len(self.train_dataloader)
        return train_loss

    def validation(self, model):
        model.eval()
        pass

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
