# -*- coding: utf-8 -*-
'''
# @Author       : Zhi
# @Date         : 2020-12-07 18:12:43
# @LastEditTime : 2020-12-07 21:50:42
'''
import os
from pathlib import Path

datasets = Path(r'/Users/liuzhi/datasets/SQuAD')
project_path = os.path.abspath(os.getcwd())

LEMMA_CACHE = f"{project_path}/cache/lemmas.feather"
VECTOR_CACHE = f"{project_path}/cache/vector.pkl"

BERT_MODEL = '/Users/liuzhi/models/bert-base-uncased'

SQuAD_URL = r'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json'
SQuAD_train = datasets / r"train-v2.0.json"
SQuAD_valid = datasets / r"dev-v2.0.json"
