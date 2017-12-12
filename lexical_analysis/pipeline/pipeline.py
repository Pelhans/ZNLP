#!/usr/bin/env python
# coding=utf-8

import pickle
import os
import sys, locale
import time
import get_pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split

sys.path.append(r'../cws_blstm/bin/') 
sys.path.append(r'../ner_blstm/bin/') 
sys.path.append(r'../pos_blstm/bin/') 

import cws_test as ct
import ner_test as nt
import pos_test as pt

cws_path = '../cws_blstm/'
ner_path = '../ner_blstm/'
pos_path = '../pos_blstm/'

class Pipeline(object):
    def __init__(self):
        print("Starting new Tensorflow session...")
        self.session = tf.Session()
        print("Loading pipeline modules...")

        self.cws_model = ct.ModelLoader(cws_path + 'ckpt/bi-lstm.ckpt-6')
        self.ner_model = nt.ModelLoader(ner_path + 'ckpt/bi-lstm.ckpt-6')
        self.pos_model = pt.ModelLoader(pos_path+ 'ckpt/bi-lstm.ckpt-6')

    def analyze(self, sentence, word2id, id2tag, zy, word2id_p, id2tag_p, word2id_n, id2tag_n):
        '''
        Return a list of three string output:cws, pos, ner
        '''

        #cws
        cws_tag = ct.cut_word(sentence ,word2id ,self.cws_model, zy)
        cws_str = " ".join(cws_tag)

        #pos
        pos_tagging = self.pos_model.predict(self.pos_model, cws_str, word2id_p, id2tag_p)

        #ner
        ner_tagging = self.ner_model.predict(self.ner_model, cws_str, word2id_n, id2tag_n)
        return cws_str, pos_tagging, ner_tagging

def get_sen(tagging):
    str = ''
    for (w, t) in tagging:
        str += u'%s/%s '%(w, t)
    return str

def main():

    word2id_c, id2tag_c, word2id_p, id2tag_p, word2id_n, id2tag_n, zy = get_pickle.get_pickle()

    pipe = Pipeline()
    #sentence = u'我爱吃北京烤鸭。'
    sentence = raw_input("请您输入一句话：").strip().decode(sys.stdin.encoding or locale.getpreferredencoding(True))
    cws, pos, ner = pipe.analyze(sentence, word2id_c, id2tag_c, zy, word2id_p, id2tag_p, word2id_n, id2tag_n)

    pos_sen = get_sen(pos)
    ner_sen = get_sen(ner)
   
    print "您输入的句子为：", sentence
    print "分词结果为：", cws
    print "词性标注结果为：", pos_sen
    print "命名实体识别结果为：", ner_sen

if __name__ == "__main__":
    main()

