#!/usr/bin/env python
# coding=utf-8

import pickle
import os

path = os.path.split(os.path.realpath(__file__))[0] +'/'

def get_pickle():
    with open(path + '../data/cws/pkl/dict.pkl', 'rb') as inpc:
        word2id_c = pickle.load(inpc)
        id2word = pickle.load(inpc)
        tag2id = pickle.load(inpc)
        id2tag_c = pickle.load(inpc)
    with open(path + '../data/cws/pkl/zy.pkl', 'rb') as inpc:
        zy = pickle.load(inpc)
    with open(path + '../data/pos/pkl/dict.pkl', 'rb') as inpp:
        word2id_p = pickle.load(inpp)
        id2word = pickle.load(inpp)
        tag2id = pickle.load(inpp)
        id2tag_p = pickle.load(inpp)
    with open(path + '../data/ner/pkl/dict.pkl', 'rb') as inpn:
        word2id_n = pickle.load(inpn)
        id2word = pickle.load(inpn)
        tag2id = pickle.load(inpn)
        id2tag_n = pickle.load(inpn)

    return word2id_c, id2tag_c, word2id_p, id2tag_p, word2id_n, id2tag_n, zy
