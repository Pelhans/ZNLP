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

path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path + r'/../../lexical_analysis/cws_blstm/bin/') 
sys.path.append(path + r'/../../lexical_analysis/pos_blstm/bin/') 

import cws_test as ct
import pos_test as pt

cws_path = path + '/../../lexical_analysis/cws_blstm/'
pos_path = path + '/../../lexical_analysis/pos_blstm/'

class Pipeline(object):
    def __init__(self):
        print("Starting new Tensorflow session...")
        self.session = tf.Session()
        print("Loading pipeline modules...")

        self.cws_model = ct.ModelLoader(cws_path + 'ckpt/bi-lstm.ckpt-6')
        self.pos_model = pt.ModelLoader(pos_path+ 'ckpt/bi-lstm.ckpt-6')

    def analyze(self, sentence, word2id, id2tag, zy, word2id_p, id2tag_p, word2id_n, id2tag_n):
        '''
        Return a list of three string output:cws, pos
        '''

        #cws
        cws_tag = ct.cut_word(sentence ,word2id ,self.cws_model, zy)
        cws_str = " ".join(cws_tag)

        #pos
        pos_tagging = self.pos_model.predict(self.pos_model, cws_str, word2id_p, id2tag_p)

        return cws_str, pos_tagging

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
    cws, pos = pipe.analyze(sentence, word2id_c, id2tag_c, zy, word2id_p, id2tag_p, word2id_n, id2tag_n)

    pos_sen = get_sen(pos)
   
    print "您输入的句子为：", sentence
    print "分词结果为：", cws
    print "词性标注结果为：", pos_sen

if __name__ == "__main__":
    main()

