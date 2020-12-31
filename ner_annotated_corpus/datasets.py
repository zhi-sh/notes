# -*- coding: utf-8 -*-
# @DateTime :2020/12/31 下午9:36
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

agg_func = lambda s: [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(), s['POS'].values.tolist(), s['Tag'].values.tolist())]


class SentenceGetter:
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        self.grouped = self.data.groupby('Sentence #').apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped[f"Sentence: {self.n_sent}"]
            self.n_sent += 1
            return s
        except:
            return None


class CustomData:
    def __init__(self, fpath, tokenizer, max_len=75):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = pd.read_csv(fpath, encoding='latin1').fillna(method='ffill')
        self.tag_values = list(set(self.data['Tag'].values))
        self.tag_values.insert(0, 'PAD')
        self.tag2idx = {t: i for i, t in enumerate(self.tag_values)}

    def get_data(self):
        getter = SentenceGetter(self.data)
        sentences = [[w[0] for w in sentence] for sentence in getter.sentences]
        labels = [[s[2] for s in sentence] for sentence in getter.sentences]
        tokenized_texts_and_labels = [self.tokenize_and_preserve_labels(sentence, labs) for sentence, labs in zip(sentences, labels)]
        tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
        labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                  maxlen=self.max_len,
                                  dtype='long',
                                  value=0.0,
                                  truncating='post',
                                  padding='post',
                                  )
        tags = pad_sequences([[self.tag2idx.get(l) for l in lab] for lab in labels],
                             maxlen=self.max_len,
                             value=self.tag2idx['PAD'],
                             padding='post',
                             dtype='long',
                             truncating='post'
                             )
        attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]
        return input_ids, attention_masks, tags

    def tokenize_and_preserve_labels(self, sentence, text_labels):
        r'''
        BERT使用BPE编码，一个单词会被拆分成多个子词，故句子需要扩展开，标签也要统一扩展
        :param tokenizer: BERT分词器
        :param sentence: 句子token列表
        :param text_labels: 句子token对应的 Tag 列表
        :return:
        '''
        tokenized_sentence = []
        labels = []
        for word, label in zip(sentence, text_labels):
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
            tokenized_sentence.extend(tokenized_word)
            labels.extend([label] * n_subwords)
        return tokenized_sentence, labels


if __name__ == '__main__':
    from settings import config
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(config.bert_model)
    dset = CustomData(f"{config.data_path}/ner_dataset.csv", tokenizer)
    input_ids, attention_mask, labels = dset.get_data()
    print(len(input_ids), len(attention_mask), len(labels))
