# -*- coding: utf-8 -*-
# @DateTime :2020/12/9 下午9:40
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
import random
import codecs
import jieba
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from vocab import Vocab


class TxtDataset(Dataset):
    def __init__(self, fpath, maxlen=512, build_vocab=False):
        data, labels, self.vocab = self._read_corpus(fpath, build_vocab)
        assert len(data) == len(labels)
        combine_array = list(zip(data, labels))
        random.shuffle(combine_array)
        self.data, self.labels = zip(*combine_array)

        self.ve = self.vocab.vocab
        self.le = self.vocab.labels
        self.maxlen = maxlen

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, ix):
        text = self.data[ix]
        label = self.labels[ix]
        return torch.tensor(self.ve.words2indices(text, self.maxlen)).long(), torch.tensor([self.le[label]]).long()

    def _read_corpus(self, fpath, build_vocab):
        root, _ = os.path.split(fpath)
        data = []
        labels = []
        with codecs.open(fpath, 'r', encoding='utf-8') as fr:
            for line in tqdm(fr):
                line = line.strip()
                pairs = line.split('\t')
                if len(pairs) != 2:
                    print(f"error line : {line}")
                    continue
                data.append(list(jieba.cut(pairs[1])))
                labels.append(pairs[0])
        if build_vocab:
            label_list = list(set(labels))
            label_map = {label: idx for idx, label in enumerate(label_list)}
            vocab = Vocab.build(data, label_map, vocab_size=50000, min_freq=2)
            vocab.save(f"{root}/vocab.json")
        else:
            vocab = Vocab.load(f"{root}/vocab.json")

        return data, labels, vocab


if __name__ == '__main__':
    td = TxtDataset(r'/Users/liuzhi/datasets/cnews/cnews.train.txt', maxlen=32, build_vocab=True)
    for t in td:
        print(t)
        break