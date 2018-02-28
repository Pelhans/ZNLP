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

"""
This script used to get word2id dict and total tags for each sentence. If you want to generate zy.pkl with ltags, please use command\
        python get_word2id.py ltags
Word dict can trans words to ids.
An example of ltags is [array([u'b', u'e', u's', u's', u'b', u'e', u's', u's', u's', u'b', u'm', u'e'], dtype='<U1')]

"""

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

#构建words 和 tags都转为id的映射
word2id = pd.Series(set_ids, index=set_words)

vocab_size = len(set_words)
ltags = df_data['tags'].values

print 'vocab_size={}'.format(vocab_size)
print 'Example of ltags: ', ltags[0:5]
print 'Example of words: ', df_data['words'].values[0]

with open('../data/word2id.pkl', 'wb') as outp:
    start = time.clock()
    pickle.dump(word2id, outp)
    if sys.argv[1] == "ltags":
        pickle.dump(ltags, outp)
    end = time.clock()
    print end-start, "s"
print 'Finished saving data....'
