#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import time
import glob
import pickle
import numpy as np
import tensorflow as tf
from model.net_work import Model
from model.get_data import BatchGenerator
from utils.get_pkl import padding
from utils.decode import decode
from configparser import ConfigParser
import argparse

path = os.path.split(os.path.realpath(__file__))[0]
parser = argparse.ArgumentParser()
parser.add_argument('--taskName', type=str, default='ner',
                    help='the lexical task name, one of "ner" "pos" "cws"')
args = parser.parse_args()

class ModelLoader(object):
    def __init__(self,ckpt_path, scope_name):
        self.cfg = ConfigParser()
        self.cfg.read(path + u'/conf/' + scope_name + '_conf.ini')
        self.scope_name = scope_name
        self.session = tf.Session()
        self.ckpt_path = ckpt_path
        self.X_inputs =  tf.placeholder(tf.int32, [None, self.cfg.get('net_work', 'timestep_size')], name='X_inputs')
        self.y_inputs = tf.placeholder(tf.int32, [None, self.cfg.get('net_work', 'timestep_size')], name='y_input')
        with tf.variable_scope(scope_name + "_blstm"):
            self._model = Model(self.cfg)
            self.cost, self.accuracy, self.correct_prediction, self.y_pred = self._model.bi_lstm(self.X_inputs, self.y_inputs)
        if len(glob.glob(self.ckpt_path + '.data*')) > 0:
            print('Loading model parameters from %s ' % ckpt_path)
            all_vars = tf.global_variables()
            model_vars = [k for k in all_vars if k.name.startswith(scope_name + "_blstm")]
            tf.train.Saver(model_vars).restore(self.session, ckpt_path)
        else:
            print('Model not found, creat with fresh parameters....')
            self.session.run(tf.global_variables_initializer())

    def predict(self, sentence, word2id, id2tag, batch_size=128):
        """首先将一个sentence根据标点和英文符号/字符串划分成多个片段text，然后对每一个片段做词性标注。"""
        if sentence:                  
            not_cuts = re.compile(u'([0-9\da-zA-Z]+)|[。，、？！.\.\?,!]')
            result = []               
            sen_part_words = []       
            sen_part_ids = []         
            len_sen = []              
            start = 0                 
            for seg_sign in not_cuts.finditer(sentence):
                words = re.split(" ", sentence[start:seg_sign.end()].strip()) if self.scope_name in ['pos', 'ner'] else sentence[start:seg_sign.end()].strip()
                sen_part_ids.append(padding(words, word2id))
                sen_part_words.append(words)
                len_sen.append(len(words))
                start = seg_sign.end() 
            total_sen_num = len(sen_part_words)
            if total_sen_num > batch_size:
                for i in range(total_sen_num/batch_size):
                    result.extend(self.tagging(sen_part_ids[i*batch_size:(i+1)*batch_size], sen_part_words[i*batch_size:(i+1)*  batch_size], id2tag, len_sen[i*batch_size:(i+1)*batch_size]))
                result.extend(self.tagging(sen_part_ids[-total_sen_num%batch_size:], sen_part_words[-total_sen_num%batch_size:  ], id2tag, len_sen[-total_sen_num%batch_size:]))
            else:                     
                result.extend(self.tagging(sen_part_ids, sen_part_words, id2tag, len_sen))
            return result

    def tagging(self, text, sen_words, id2tag, len_sen):
        if text:
            max_len = self.cfg.getint('get_pkl', 'max_len')
            y_pred = []
            tags = []
            text_len = len(text)
            if text_len == 1:
                X_batch = [text[0]]
            else:
                X_batch = np.squeeze(text, axis=(1,))
            fetches = [self.y_pred]
            feed_dict = {self.X_inputs:X_batch, self._model.lr:0.0, self._model.batch_size:len(text), self._model.keep_prob:1.0}
            _y_pred = self.session.run(fetches, feed_dict)
            _y_pred = np.squeeze(_y_pred, axis=(0,))
            for i in range(text_len):
                tags.append(decode(self.cfg, _y_pred[i*max_len:i*max_len + len_sen[i]], id2tag))
            return zip(sen_words, tags)

def show_result(tags):          
    str = ''                    
    for (w, t) in tags:         
        for i in range(len(t)): 
            str += u'%s/%s '%(w[i], t[i])
    return str

def main():
    cfg = ConfigParser()
    cfg.read(path + u'/conf/' + args.taskName + '_conf.ini')
    ckpt_path = cfg.get('file_path', 'model') + '-6'
    model = ModelLoader(ckpt_path, args.taskName)

    start = time.clock()
    with open(cfg.get('file_path', 'dict'), 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)

    sentence = u'我 爱 吃 北京 烤鸭 。北京 烤鸭 真 好吃 。'
#    sentence = u'人们思考问题往往不是从零开始的。就好像你现在阅读这篇文章一样。'
    tagging = model.predict(sentence, word2id, id2tag)

    norm_result = (tagging)   
    print(args.taskName + " 标记结果为: \n", norm_result)
    print(time.clock() - start , " s")


if __name__ == '__main__':
    main()
