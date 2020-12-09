# -*- coding: utf-8 -*-
# @DateTime :2020/12/9 下午9:35
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import json
from itertools import chain
from collections import Counter


class VocabEntry:
    def __init__(self, word2idx=None):
        if word2idx:
            self.word2idx = word2idx
        else:
            self.word2idx = dict()
            self.word2idx['<PAD>'] = 0
            self.word2idx['<UNK>'] = 1
        self.unk_id = self.word2idx['<UNK>']
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __getitem__(self, word):
        return self.word2idx.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2idx

    def __setitem__(self, key, value):
        raise ValueError('Vocab is readonly!')

    def __len__(self):
        return len(self.word2idx)

    def __repr__(self):
        return f"Vocab=[size={len(self.word2idx)}]"

    def add(self, word):
        if word not in self.word2idx:
            wid = self.word2idx[word] = len(self.word2idx)
            self.idx2word[wid] = word
            return wid
        else:
            return self.word2idx[word]

    def words2indices(self, sents, maxlen=512):
        idxs = [self.word2idx.get(w, self.unk_id) for w in sents]
        length = len(idxs)
        if length >= maxlen:
            return idxs[:maxlen]
        else:
            return idxs + [self.word2idx['<PAD>']] * (maxlen - length)

    def indices2words(self, idxs):
        return [self.word2idx[i] for i in idxs]

    @staticmethod
    def from_corpus(corpus, size, min_freq=3):
        r'''从给定语料中创建 Vocab'''
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = word_freq.most_common(size - 2)
        valid_words = [word for word, value in valid_words if value >= min_freq]
        print(f"number of word types: {len(word_freq)}, number of wword types w/frequency>={min_freq}:{len(valid_words)}")
        for word in valid_words:
            vocab_entry.add(word)
        return vocab_entry


class Vocab:
    def __init__(self, src_vocab: VocabEntry, labels: dict):
        self.vocab = src_vocab
        self.labels = labels

    @staticmethod
    def build(src_sents, labels, vocab_size, min_freq):
        print('initialize source vocabulary...')
        src = VocabEntry.from_corpus(src_sents, vocab_size, min_freq)

        return Vocab(src, labels)

    def save(self, fpath):
        with open(fpath, 'w', encoding='utf-8') as fw:
            json.dump(dict(word2idx=self.vocab.word2idx, labels=self.labels), fw, ensure_ascii=False, indent=2)

    @staticmethod
    def load(fpath):
        with open(fpath, encoding='utf-8') as fr:
            entry = json.load(fr)
            word2idx = entry['word2idx']
            labels = entry['labels']

        return Vocab(VocabEntry(word2idx), labels)

    def __repr__(self):
        return f"Vocab(source {len(self.vocab)} words)"
