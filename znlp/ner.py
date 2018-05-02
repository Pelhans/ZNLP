#!/usr/bin/env python
# coding=utf-8
      
import os,sys
import glob    
import pickle    
import tensorflow as tf    
from sklearn.model_selection import train_test_split
      
path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path + r'/../lexical_analysis/ner_blstm/bin/')
import ner_blstm as nb
import ner_tags
      
config = nb.config_ch()

class load_Model():
                                                
    global model, word2id, id2tag
    ckpt_path = path + r'/../lexical_analysis/ner_blstm/ckpt/bi-lstm.ckpt-6'
    model = ner_tags.ModelLoader(ckpt_path)
                    
    with open(path + r'/../lexical_analysis/ner_blstm/data/pkl/dict_data.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
      
def ntag(sentence):
    global model, word2id, id2tag
    return model.predict(model, sentence, word2id, id2tag)
      
def show_result(tags):
    str = ''
    for (w, t) in tags:
        for i in range(len(t)):
            str += u'%s/%s '%(w[i], t[i])
    print "NER标记结果为: \n", str

def main():
    load_Model()
    sentence = u'我 爱 吃 北京 烤鸭 。'
    result = ntag(sentence)
    show_result(result)
           
if __name__ == '__main__':
    main() 

