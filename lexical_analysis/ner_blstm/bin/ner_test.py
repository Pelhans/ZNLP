#!/usr/bin/env python
# coding=utf-8

import re
import time
import glob
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import ner_blstm as nb
from sklearn.model_selection import train_test_split

config = nb.config_ch()
class ModelLoader(object):
    def __init__(self,ckpt_path):
        self.session = tf.Session()
        self.ckpt_path = ckpt_path
        self.X_inputs =  tf.placeholder(tf.int32, [None, config.timestep_size], name='X_inputs')
        self.y_inputs = tf.placeholder(tf.int32, [None, config.timestep_size], name='y_input')
        #with tf.variable_scope('ner_embedding'):
        #    with tf.device("/cpu:0"):
        #        embedding = tf.get_variable("ner_embedding", [config.vocab_size, config.embedding_size], dtype=tf.float32)
        with tf.variable_scope('ner_blstm'):
            self.cost, self.accuracy, self.correct_prediction, self.y_pred = nb.bi_lstm(self.X_inputs, self.y_inputs)
        if len(glob.glob(self.ckpt_path + '.data*')) > 0:
            print 'Loading model parameters from %s ' % ckpt_path
            all_vars = tf.global_variables()
            model_vars = [k for k in all_vars if k.name.startswith("ner_blstm")]
            tf.train.Saver(model_vars).restore(self.session, ckpt_path)
        else:
            print 'Model not found, creat with fresh parameters....'
            self.session.run(tf.global_variables_initializer())

    def predict(self, model, sentence, word2id, id2tag):
        if sentence:
            words = list(re.split(" ", sentence))
            text_len = len(words)
            X_batch = text2ids(sentence, word2id)
            fetches = [model.y_pred]
            feed_dict = {model.X_inputs:X_batch, config.lr:1.0, config.batch_size:1, config.keep_prob:1.0}
            _y_pred = model.session.run(fetches, feed_dict)[0][:text_len]
            y_pred = tf.cast(tf.argmax(_y_pred, 1), tf.int32)
            l_tmp = list(model.session.run(y_pred))
            tags = list(id2tag[l_tmp])
            return zip(words, tags)


def text2ids(sentence, word2id):
    """把词片段text转为 ids."""
    words = re.split(" ", sentence)
    ids = list(word2id[word] if word in word2id.index else word2id["UNK"] for word in words)
    if len(ids) >= config.max_len:  # 长则弃掉
        print u'ner 输出片段超过%d部分无法处理' % (config.max_len) 
        return ids[:config.max_len]
    ids.extend([0]*(config.max_len-len(ids))) # 短则补全
    ids = np.asarray(ids).reshape([-1, config.max_len])
    return ids

def main():
    ckpt_path = '../ckpt/bi-lstm.ckpt-6'
    model = ModelLoader(ckpt_path)

    with open('../data/data.pkl', 'rb') as inp:
        X = pickle.load(inp)
        y = pickle.load(inp)
        ltags = pickle.load(inp)
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)

    sentence = u'我 爱 吃 北京 烤鸭 。'
    tagging = model.predict(model, sentence, word2id, id2tag)

    str = ''
    for (w, t) in tagging:
        str += u'%s/%s '%(w, t)
    print "NER标记结果为:"
    print str

if __name__ == '__main__':
    main()
