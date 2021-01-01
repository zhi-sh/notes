# -*- coding: utf-8 -*-
# @DateTime :2020/12/31 下午9:22
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import torch


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


# 项目配置
config = Config(
    # 数据集参数
    data_path=r'/Users/liuzhi/datasets/annotated_corpus_4_ner',

    # 模型参数
    bert_model=r'/Users/liuzhi/models/torch/bert-base-uncased',
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    num_labels=18,  # Tag类别，通过datasets类可获取

    # 训练参数
    epochs=1,
    bs=32,
    lr=3e-5,
    eps=1e-8,
)
