# -*- coding: utf-8 -*-
'''
# @Author       : Zhi
# @Date         : 2020-12-07 18:12:43
# @LastEditTime : 2020-12-07 20:44:07
'''
from pathlib import Path

datasets = Path(r'/Users/liuzhi/datasets/SQuAD')
SQuAD_URL = r'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json'
SQuAD_train = datasets / r"train-v2.0.json"
SQuAD_valid = datasets / r"dev-v2.0.json"
