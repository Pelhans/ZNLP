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

DIR=os.getcwd()

if not os.path.exists('../data/'):
    os.makedirs('../data/')

with open("../data/msr_train.txt", "rb") as inp:
    texts = inp.read().decode('gbk')
sentences = texts.split('\r\n') #根据换行符对文本进行切分

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

sentences = re.split(u'[，。！？、‘’“”]/[bems]', texts)
print 'Sentences number:', len(sentences)
print 'Sentence Example: \n', sentences[1]

def get_Xy(sentence):
    #将sentences处理成[word],[tag]的形式
    word_tags = re.findall('(.)/(.)',sentence)
    if word_tags:
        word_tags = np.asarray(word_tags)
        words = word_tags[:, 0]
        tags = word_tags[:, 1]
        return words, tags
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

#统计所有word
sr_allwords = pd.Series(all_words)
sr_allwords = sr_allwords.value_counts()
set_words = sr_allwords.index
set_ids = range(1, len(set_words) + 1)
tags = ['x', 's', 'b', 'm', 'e']
tag_ids = range(len(tags))

#构建words 和 tags都转为id的映射
word2id = pd.Series(set_ids, index=set_words)
id2word = pd.Series(set_words, index = set_ids)
tag2id = pd.Series(tag_ids, index = tags)
id2tag = pd.Series(tags, index = tag_ids)
vocab_size = len(set_words)
print 'vocab_size={}'.format(vocab_size)

max_len = 50
def X_padding(words):
    #把words转为id形式，并自动补全为max_len长度
    ids = list(word2id[words])
    if len(ids) >= max_len:
        return ids[:max_len]
    ids.extend([0]*(max_len - len(ids)))
    return ids

def y_padding(tags):
    #把tag转为id形式，并自动补全为max_len长度
    ids = list(tag2id[tags])
    if len(ids) >= max_len:
        return ids[:max_len]
    ids.extend([0]*(max_len - len(ids)))
    return ids

start = time.clock()
df_data['X'] = df_data['words'].apply(X_padding)
df_data['y'] = df_data['tags'].apply(y_padding)
end = time.clock()
print end-start

X = np.asarray(list(df_data['X'].values))
y = np.asarray(list(df_data['y'].values))
print 'X.shape={}, y.shape={}'.format(X.shape, y.shape)
print 'Example of words: ', df_data['words'].values[0]
print 'Example of X: ', X[0]
print 'Example of tags: ', df_data['tags'].values[0:5]
print 'Example of y: ', y[0]

with open('../data/train_data.pkl', 'wb') as outp:
    start = time.clock()
    pickle.dump(X, outp)
    pickle.dump(y, outp)
    end = time.clock()
    print end-start, "s"
    outp.close()

del X, y

ltags = df_data['tags'].values
print 'Example of ltags: ', ltags[0:5]
with open('../data/ltags_data.pkl', 'wb') as outp:
    pickle.dump(ltags, outp)
    start = time.clock()
    end = time.clock()
    print end-start, "s"
    outp.close()

del df_data, ltags

with open('../data/dict_data.pkl', 'wb') as outp:
    start = time.clock()
    pickle.dump(word2id, outp)
    pickle.dump(id2word, outp)
    pickle.dump(tag2id, outp)
    pickle.dump(id2tag, outp)
    end = time.clock()
    print end-start, "s"
    outp.close()

print 'Finished saving data....'
