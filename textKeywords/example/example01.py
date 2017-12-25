#-*- encoding:utf-8 -*-
from __future__ import print_function

import sys
import os
sys.path.append(r'../bin/')
sys.path.append(r'../util/')
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

import codecs

path = os.path.split(os.path.realpath(__file__))
path_pip = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path_pip + r'/../pipeline')
import pipeline, get_pickle                                                                           
              
pic = get_pickle.get_pickle()  
pipe = pipeline.Pipeline()

import TextRank4Keyword, TextRank4Sentence

text = codecs.open('../data/01.txt', 'r', 'utf-8').read()
print (os.path.realpath(__file__))
tr4w = TextRank4Keyword.TextRank4Keyword()

tr4w.analyze(pic, pipe, text=text, lower=True, window=2)   # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象

print( '关键词：' )
for item in tr4w.get_keywords(20, word_min_len=1):
    print(item.word, item.weight)

print()
print( '关键短语：' )
for phrase in tr4w.get_keyphrases(keywords_num=20, min_occur_num= 2):
    print(phrase)

tr4s = TextRank4Sentence.TextRank4Sentence()
tr4s.analyze(pic, pipe, text=text, lower=True, source = 'all_filters')

print()
print( '摘要：' )
for item in tr4s.get_key_sentences(num=3):
    print(item.index, item.weight, item.sentence)
