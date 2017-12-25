#!/usr/bin/env python
# coding=utf-8

import re
import os
import sys
import time
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from itertools import chain

file = "dev_ner.txt" if len(sys.argv)==1 else sys.argv[1]

with open("../data/" + str(file), "rb") as inp:
    texts = inp.read().decode('utf-8')
sentences = texts.split('\n') #根据换行符对文本进行切分

with open('../data/data.pkl', 'rb') as inp:
    X = pickle.load(inp)
    y = pickle.load(inp)
    ltags = pickle.load(inp)
    word2id = pickle.load(inp)

def clean(s): #将句子中如开头和中间无匹配的引号去掉
    if u'“/s' not in s:
        return s.replace(u' ”/s', '')
    elif u'”/s' not in s:
        return s.replace(u'“/s ', '')
    elif u'‘/s' not in s:
        return s.replace(u' ’/s', '')
    elif u'’/s' not in s:
        return s.replace(u'‘/s ', '')
    else:
        return s

texts = u''.join(map(clean, sentences))
print 'Length of text is %d' % len(texts)
print 'Example of texts: \n', texts[:300]

def get_Xy(sentence):
    #将sentences处理成[word],[tag]的形式
    new_word = []
    new_tag = []
    words = re.split("\s+", sentence)
    if words:
        for word in words:
            pairs = word.split("/")
            if len(pairs) == 2:
                if (len(pairs[0].strip())!=0 and len(pairs[1].strip())!=0):
                    new_word.append(pairs[0])
                    new_tag.append(pairs[1])
        return new_word, new_tag
    return None

datas = list()
labels = list()
print 'Start creating words and tag data....'
for sentence in tqdm(iter(sentences)):  #need tqdm
    result = get_Xy(sentence)
    if result:
        datas.append(result[0])
        labels.append(result[1])
print 'Length of data is %d' % len(datas)
print 'Example of datas: ', datas[0]
print 'Example of labels:', labels[0]

df_data = pd.DataFrame({'words': datas, "tags": labels}, index = range(len(datas)))
df_data['sentence_len'] = df_data['words'].apply(lambda words: len(words))
df_data.head(2)

#使用 chain(*list)函数把多个list拼接起来
all_words = list(chain(*df_data['words'].values))
all_words.append(u'UNK')
print all_words[0:10]

#统计所有word
sr_allwords = pd.Series(all_words)
sr_allwords = sr_allwords.value_counts()
set_words = sr_allwords.index
print "set_words: ", list(set_words)[:10]
set_ids = list(word2id[word] if word in word2id.index else word2id["UNK"] for word in set_words)
tags = ['nz', 'nt', 'ns', 'nr', 'nan']
tag_ids = range(len(tags))

#构建words 和 tags都转为id的映射
word2id_test = pd.Series(set_ids, index=set_words)
id2word_test = pd.Series(set_words, index = set_ids)
tag2id_test = pd.Series(tag_ids, index = tags)
id2tag_test = pd.Series(tags, index = tag_ids)
vocab_size = len(set_words)
print 'vocab_size={}'.format(vocab_size)

max_len = 50
def X_padding(words):
    #把words转为id形式，并自动补全为max_len长度
    ids = list(word2id_test[words])
    if len(ids) >= max_len:
        return ids[:max_len]
    ids.extend([0]*(max_len - len(ids)))
    return ids

def y_padding(tags):
    #把tag转为id形式，并自动补全为max_len长度
    ids = list(tag2id_test[tags])
    if len(ids) >= max_len:
        return ids[:max_len]
    ids.extend([0]*(max_len - len(ids)))
    return ids

start = time.clock()
df_data['X'] = df_data['words'].apply(X_padding)
df_data['y'] = df_data['tags'].apply(y_padding)
end = time.clock()
print end-start

X_test = np.asarray(list(df_data['X'].values))
y_test = np.asarray(list(df_data['y'].values))
ltags_test = df_data['tags'].values
print 'X.shape={}, y.shape={}'.format(X_test.shape, y_test.shape)
print 'Example of words: ', df_data['words'].values[0]
print 'Example of X: ', X_test[0]
print 'Example of tags: ', df_data['tags'].values[0:5]
print 'Example of ltags: ', ltags_test[0:5]
print 'Example of y: ', y_test[0]
print 'Eaxmple of ltags: ', ltags_test[0], ltags.shape

with open('../data/data_test.pkl', 'wb') as outp:
    start = time.clock()
    pickle.dump(X_test, outp)
    pickle.dump(y_test, outp)
    pickle.dump(ltags_test, outp)
    pickle.dump(word2id_test, outp)
    pickle.dump(id2word_test, outp)
    pickle.dump(tag2id_test, outp)
    pickle.dump(id2tag_test, outp)
    end = time.clock()
    print end-start
print 'Finished saving data....'
