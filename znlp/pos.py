#!/usr/bin/env python
# coding=utf-8

import os,sys
import glob                                     
import pickle                                   
import tensorflow as tf                         
from sklearn.model_selection import train_test_split

path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path + r'/../lexical_analysis/pos_blstm/bin/')
import pos_blstm as pb
import pos_tags

config = pb.config_ch()

class load_Model():

    global model, word2id, id2tag
    ckpt_path = path + r'/../lexical_analysis/pos_blstm/ckpt/bi-lstm.ckpt-6'
    model = pos_tags.ModelLoader(ckpt_path)
                 
    with open(path + r'/../lexical_analysis/pos_blstm/data/pkl/dict_data.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)

def ptag(sentence):
    global model, word2id, id2tag
    return model.predict(model, sentence, word2id, id2tag)

def show_result(tags):
    str = ''       
    for (w, t) in tags:
        for i in range(len(t)):
            str += u'%s/%s '%(w[i], t[i])
    print "POS标记结果为: \n", str 

def main():
    
    load_Model()
    sentence = u'词法 分析 终于 完成 了 。 这 都 叫 啥 事 啊 。'
    result = ptag(sentence)
    show_result(result)

if __name__ == '__main__':
    main()
