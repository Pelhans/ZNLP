#!/usr/bin/env python
# coding=utf-8

import time, sys,os
import pickle
import tensorflow as tf
 
path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path + r'/../lexical_analysis/cws_blstm/bin/')
import cws_blstm as cb
import cut_words

config = cb.config_ch()
batch_size = 128     
wget_zy = 'not_get_zy'
if len(sys.argv) == 1:
    print "Using default config. batch_size = 128 and not_get_zy."
elif len(sys.argv) == 3:
    batch_size = sys.argv[1]
    wget_zy = sys.argv[2]


class load_Model():

    global model, word2id, zy

    ckpt_path = path + '/../lexical_analysis/cws_blstm/ckpt/bi-lstm.ckpt-6'
    model = cut_words.ModelLoader(ckpt_path)
      
    with open(path + '/../lexical_analysis/cws_blstm/data/word2id.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        if wget_zy == "get_zy":
            ltags = pickle.load(inp)
      
    if wget_zy == "get_zy":
        get_zy(ltags)  #这行用来生成转移概率的pkl文件，一次生成后就可以注释掉了
      
    with open(path + '/../lexical_analysis/cws_blstm/data/zy.pkl', 'rb') as inp:
        zy = pickle.load(inp)
      
      
def cut(sentence):
    global model, word2id, zy
    return cut_words.cut_word(sentence ,word2id ,model, zy, batch_size)

def show_result(result):
    rss = ''  
    for each in result:
        if isinstance(each, list):
            each = "".join(each)
        rss = rss + each + ' / '
    print rss


def main():
 
    sentence =                                                                                                            u'人们思考问题往往不是从零开始的。就好像你现在阅读这篇文章一样，你对每个词的理解都会依赖于你前面看到的一些词，\
而不是把你前面看的内容全部抛弃了，忘记了，再去理解这个单词。也就是说，人们的思维总是会有延续性的。'
    start = time.clock()
    mcws = load_Model()
    result = cut(sentence)
    show_result(result)
    print time.clock() - start, "s"
 
if __name__ == '__main__':
    main()

