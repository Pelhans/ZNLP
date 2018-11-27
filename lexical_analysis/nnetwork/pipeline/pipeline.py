#!/usr/bin/env python
# coding=utf-8

import pickle
import os
import sys, locale
import time
import get_pickle
import numpy as np
import tensorflow as tf
from configparser import ConfigParser

#cfg = ConfigParser()
#cfg.read(u'conf/ner_conf.ini')

path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path + r'/../')
from predict import ModelLoader
from predict import show_result

class Pipeline(object):
    def __init__(self):
        print("Starting new Tensorflow session...")
        self.session = tf.Session()
        print("Loading pipeline modules...")

        self.cws_model = ModelLoader(r'../ckpt/cws/bi-lstm.ckpt-6', 'cws')
        self.pos_model = ModelLoader(r'../ckpt/pos/bi-lstm.ckpt-6', 'pos')
        self.ner_model = ModelLoader(r'../ckpt/ner/bi-lstm.ckpt-6', 'ner')

    def analyze(self, sentence, word2id, id2tag, zy, word2id_p, id2tag_p, word2id_n, id2tag_n):
        '''
        Return a list of three string output:cws, pos, ner
        '''

        #cws
        cws_tag = self.cws_model.predict(sentence ,word2id , id2tag)
        cws_str = merge_cws(cws_tag)

        #pos
        pos_tagging = self.pos_model.predict(cws_str, word2id_p, id2tag_p)

        #ner
        ner_tagging = self.ner_model.predict(cws_str, word2id_n, id2tag_n)
        return cws_str, pos_tagging, ner_tagging

def merge_cws(cws_tag):
    words = []
    tmp = []
    rss = ''
    for (w, t) in cws_tag:
        for i in range(len(t)):
            if t[i] in ['s', 'n']:
                words.append(w[i])
            else:
                tmp.extend(w[i])
                if t[i] == 'e':
                    words.append(tmp)
                    tmp = []
    for each in words:
        if isinstance(each, list):
            each = "".join(each)
        rss += each + ' '
    return rss

def main():

    word2id_c, id2tag_c, word2id_p, id2tag_p, word2id_n, id2tag_n, zy = get_pickle.get_pickle()

    pipe = Pipeline()
    sentence = u'我爱吃北京烤鸭。我爱中华人民共和国。'
#    sentence = raw_input("请您输入一句话：").strip().decode(sys.stdin.encoding or locale.getpreferredencoding(True))
    cws, pos, ner = pipe.analyze(sentence, word2id_c, id2tag_c, zy, word2id_p, id2tag_p, word2id_n, id2tag_n)

    pos_sen = show_result(pos)
    ner_sen = show_result(ner)
   
    print "您输入的句子为：", sentence
    print "分词结果为：", cws
    print "词性标注结果为：", pos_sen
    print "命名实体识别结果为：", ner_sen

if __name__ == "__main__":
    main()

