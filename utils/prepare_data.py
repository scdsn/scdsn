# coding=utf-8
"""
Author  : Jane
Contact : xijian@ict.ac.cn
Time    : 2021/3/16 16:11
Desc:
"""

import pandas as pd
import re
import jieba
from collections import Counter
from tqdm import tqdm
import pickle
import os
import numpy as np

import torch
import torchtext
import pandas as pd
import torchtext.data as data

import sys
# sys.path.append('/home/xijian/pycharm_projects/document-level-classification/')
from bert_document_classification.config import *

df_stopwords = pd.read_csv(stopwords_file, index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='utf-8')
STOPWORDS_SET = set(df_stopwords['stopword'].values)


# 读取数据  数据格式：content    label
def read_data(filepath):
    df_data = pd.read_csv(filepath, encoding='UTF-8', sep='\t', names=['label', 'content'], index_col=False)
    df_data = df_data.dropna()
    print(df_data.head())

    # x_data, y_data = df_data['content'][:100], df_data['label'][:100] # 用于测试功能
    x_data, y_data = df_data['content'], df_data['label']
    print('*'*27, x_data.shape, len(x_data[0]), y_data.shape)  # (50000,) 746 (50000,)
    print(label2id)
    y_data = [label2id[y] for y in y_data]
    # y_data = torch.tensor(y_data, dtype=torch.long)

    return x_data, y_data


# 
def clear_text(text):
    p = re.compile(r"[^\u4e00-\u9fa5^0-9^a-z^A-Z\-、，。！？：；（）《》【】,!\?:;[\]()]")  # 匹配不是中文、数字、字母、短横线的部分字符
    return p.sub('', text)  # 将text中匹配到的字符替换成空字符


# 分词
def tokenize(text):
    text = clear_text(text)
    segs = jieba.lcut(text.strip(), cut_all=False)  # cut_all=False是精确模式，True是全模式；默认模式是False 返回分词后的列表
    segs = filter(lambda x: len(x.strip()) > 1, segs)  # 词长度要>1，没有保留标点符号

    global STOPWORDS_SET
    segs = filter(lambda x: x not in STOPWORDS_SET, segs) # 去除停用词 segs是一个filter object
    return list(segs)


# 只分句
def do_seg_sentences(doc):
    # sents = re.split(r'，|。|！|？|：|；|,|!|\?|:|;', doc)
    sents = re.split(r'，|。|！|？|,|!|\?', doc)
    sentences = [s for s in sents if len(s.strip()) != 0]
    return sentences


# 过滤低频词
def filter_lowfreq_words(arr, vocab):
    # arr是一个batch，以list的形式出现，list长度=batchsize，list中每个元素是长度=MAX_LEN的句子，句子已经分词，词已经转化为index
    arr = [[x if x < total_words else 0 for x in example] for example in arr]  # 词的ID是按频率降序排序的 <unk>=0
    return arr


# 顺序：tokenize分词，preprocessing，建立词表build vocab，batch（padding & truncate to maxlen），postprocessing
NESTED = torchtext.data.Field(tokenize=tokenize,
                              sequential=True,
                              fix_length=sent_maxlen,
                              postprocessing=filter_lowfreq_words) # after numericalizing but before the numbers are turned into a Tensor)
TEXT = torchtext.data.NestedField(NESTED,
                            fix_length=doc_maxlen,
                            tokenize=do_seg_sentences,
                            )
LABEL = torchtext.data.Field(sequential=False,
                             use_vocab=False
                             )



# 定义字段
# TEXT = torchtext.data.Field(sequential=True, tokenize=tokenize, lower=True, batch_first=True)
# LABEL = torchtext.data.Field(sequential=False, use_vocab=False, is_target=True)
FEATURE = torchtext.data.Field(sequential=False, use_vocab=False, dtype=torch.float)

def get_dataset(x_data, y_data, x2_data):
    fields = [('inp', TEXT), ('lab', LABEL), ('fin', FEATURE)]  # filed信息 fields dict[str, Field])
    examples = []  # list(Example)
    for inp, lab, fin in tqdm(zip(x_data, y_data, x2_data)): # 进度条
        # 创建Example时会调用field.preprocess方法
        examples.append(torchtext.data.Example.fromlist([inp, lab, fin], fields))
    return examples, fields

class DataLoader:
    def __init__(self, data_iter):
        self.data_iter = data_iter
        self.length = len(data_iter)  # 一共有多少个batch？

    def __len__(self):
        return self.length

    def __iter__(self):
        # 注意，在此处调整text的shape为batch first，并调整label的shape和dtype
        for batch in self.data_iter:
            yield (batch.inp, batch.lab.long(), batch.fin)  # label->long

new_path = r'C:\Users\FxxkDatabase\Desktop\document-level-classification-master\document-level-classification'

save_dir = './save/'
imgs_dir = './imgs/'
stopwords_file = os.path.join(new_path, 'data/zh_data/stopwords.txt')
vocab_path = r'C:\Users\FxxkDatabase\Desktop\bert_document_classification-master\bert_document_classification-master\bert_document_classification\tokenizer\vocab.pkl'

def load_data(x1_data, x2_data, y_data, traindata=False, shuffle=False):
    # 创建TabularDataset
    x2_numeric_data = x2_data.values.tolist()  # 转换为列表
    # 创建 Field 对象用于数值字段
    NUMERIC_FIELD = data.Field(sequential=False, use_vocab=False, dtype=torch.float)

    # 将数值字段加入到 fields 列表中
    fields = [('inp', TEXT), ('fin', NUMERIC_FIELD), ('lab', LABEL)]

    # 创建 Example
    examples = []
    i = 0
    for x1, x2, y in tqdm(zip(x1_data, x2_numeric_data, y_data)):
        # print(x2)
        i += 1
        examples.append(data.Example.fromlist([x1, x2, y], fields))
        # if i == 100:
        #     break
    ds = torchtext.data.Dataset(examples, fields)
    print('*'*27, len(ds[0].inp), len(ds[1].inp), ds[0].inp, ds[0].lab, ds[0].fin) # 还是汉字，还未ID化

    # 保存 Dataset 对象到本地文件
    torch.save(ds.examples, 'dataset_examples.pt')
    torch.save(ds.fields, 'dataset_fields.pt')

    # 构建词汇表
    print('vocab_path', vocab_path)
    if os.path.exists(vocab_path):
        print('词表存在!')
        with open(vocab_path, 'rb') as handle:
            c = pickle.load(handle)
        TEXT.vocab = torchtext.vocab.Vocab(c, max_size=total_words)
        NESTED.vocab = torchtext.vocab.Vocab(c, max_size=total_words)
    else:
        print('词表不存在!')
        TEXT.build_vocab(ds, max_size=total_words)
        with open(vocab_path, 'wb') as handle:
            pickle.dump(TEXT.vocab.freqs, handle)

    # 创建数据加载器
    ds_iter = torchtext.data.Iterator(ds,
                                      batch_size=batch_size,
                                      device='cuda:0',
                                      train=traindata,
                                      shuffle=shuffle,
                                      sort=False)

    # return ds_iter
    data_loader = DataLoader(ds_iter)
    return data_loader



if __name__=='__main__':
    train_dataloader = load_data(data_base_dir + 'cnews.train.txt', traindata=True, shuffle=True)
    val_dataloader = load_data(data_base_dir + 'cnews.val.txt', traindata=False, shuffle=False)

    print('*' * 27, 'len(train_dataloader):', len(train_dataloader))  # 1000 个step/batch
    for batch_text, batch_label in train_dataloader:
        print(batch_text.shape, batch_label.shape)  # [b,100,10], [b]
        # print(batch_text[0])
        print(batch_label[0], batch_label[0].dtype)  # tensor(5) torch.int64
        break


