#!/usr/bin/env python
# coding=utf-8

import pickle
import os

path = os.path.split(os.path.realpath(__file__))[0] +'/'
cws_path = path + '../../lexical_analysis/cws_blstm/'
ner_path = path + '../../lexical_analysis/ner_blstm/'
pos_path = path + '../../lexical_analysis/pos_blstm/'

def get_pickle():
    with open(cws_path + 'data/data.pkl', 'rb') as inpc:
        X = pickle.load(inpc)
        y = pickle.load(inpc)
        ltags = pickle.load(inpc)
        word2id_c = pickle.load(inpc)
        id2word = pickle.load(inpc)
        tag2id = pickle.load(inpc)
        id2tag_c = pickle.load(inpc)
    with open(pos_path + 'data/data.pkl', 'rb') as inpp:
        X = pickle.load(inpp)
        y = pickle.load(inpp)
        ltags = pickle.load(inpp)
        word2id_p = pickle.load(inpp)
        id2word = pickle.load(inpp)
        tag2id = pickle.load(inpp)
        id2tag_p = pickle.load(inpp)
    with open(ner_path + 'data/data.pkl', 'rb') as inpn:
        X = pickle.load(inpn)
        y = pickle.load(inpn)
        ltags = pickle.load(inpn)
        word2id_n = pickle.load(inpn)
        id2word = pickle.load(inpn)
        tag2id = pickle.load(inpn)
        id2tag_n = pickle.load(inpn)

    with open(cws_path + 'data/zy.pkl', 'rb') as inpz:
        zy = pickle.load(inpz)

    return word2id_c, id2tag_c, word2id_p, id2tag_p, word2id_n, id2tag_n, zy
