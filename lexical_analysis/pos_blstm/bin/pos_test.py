#!/usr/bin/env python
# coding=utf-8

import re
import time
import glob
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import pos_blstm as pb
from sklearn.model_selection import train_test_split

config = pb.config_ch()
class ModelLoader(object):
    def __init__(self,ckpt_path):
        self.session = tf.Session()
        self.ckpt_path = ckpt_path
        self.X_inputs =  tf.placeholder(tf.int32, [None, config.timestep_size], name='X_inputs')
        self.y_inputs = tf.placeholder(tf.int32, [None, config.timestep_size], name='y_input')
            
        with tf.variable_scope('pos_blstm'):
            self.cost, self.accuracy, self.correct_prediction, self.y_pred = pb.bi_lstm(self.X_inputs, self.y_inputs)

        if len(glob.glob(self.ckpt_path + '.data*')) > 0:
            print 'Loading model parameters from %s ' % ckpt_path
            all_vars = tf.global_variables()
            model_vars = [k for k in all_vars if k.name.startswith("pos_blstm")]
            tf.train.Saver(model_vars).restore(self.session, ckpt_path)
        else:
            print 'Model not found, creat with fresh parameters....'
            self.session.run(tf.global_variables_initializer())

    def predict(self, model, sentence, word2id, id2tag, batch_size=128):
        """首先将一个sentence根据标点和英文符号/字符串划分成多个片段text，然后对每一个片段做词性标注。"""
        if sentence:
            not_cuts = re.compile(u'([0-9\da-zA-Z]+)|[。，、？！.\.\?,!]')
           
            result = []
            sen_part_words = []
            sen_part_ids = []
            len_sen = []
            start = 0

            for seg_sign in not_cuts.finditer(sentence):
                sen_part_ids.append(text2ids(sentence[start:seg_sign.end()].strip(), word2id))
                sen_part_words.append(re.split(" ", sentence[start:seg_sign.end()].strip()))
                len_sen.append(len(re.split(" ", sentence[start:seg_sign.end()].strip())))
                start = seg_sign.end()

            total_sen_num = len(sen_part_words)
            print "total_sen_num: ", total_sen_num

            if total_sen_num > batch_size:
                for i in range(total_sen_num/batch_size):
                    result.extend(tag_pos(sen_part_ids[i*batch_size:(i+1)*batch_size], sen_part_words[i*batch_size:(i+1)*batch_size], id2tag, model, len_sen[i*batch_size:(i+1)*batch_size]))
                result.extend(tag_pos(sen_part_ids[-total_sen_num%batch_size:], sen_part_words[-total_sen_num%batch_size:], id2tag, model, len_sen[-total_sen_num%batch_size:]))
            else:
                result.extend(tag_pos(sen_part_ids, sen_part_words, id2tag, model, len_sen))
            return result

def tag_pos(text, sen_words, id2tag, model, len_sen):
    if text:
        y_pred = []
        text_len = len(text)

        if text_len == 1:
            X_batch = text[0]
        else:
            X_batch = np.squeeze(text, axis=(1,))
        fetches = [model.y_pred]
        feed_dict = {model.X_inputs:X_batch, config.lr:1.0, config.batch_size:len(text), config.keep_prob:1.0}
        _y_pred = model.session.run(fetches, feed_dict)
        _y_pred = np.squeeze(_y_pred, axis=(0,))
        for i in range(text_len):
            y_pred.append(tf.cast(tf.argmax(_y_pred[i*config.max_len:i*config.max_len + len_sen[i]], 1), tf.int32))
        l_tmp = list(model.session.run(y_pred))
        tags = [list(id2tag[t_tmp]) for t_tmp in l_tmp]

        return zip(sen_words, tags)



def text2ids(sentence, word2id):
    """把词片段text转为 ids."""
    words = re.split(" ", sentence)
    ids = list(word2id[word] if word in word2id.index else word2id["UNK"] for word in words)
    if len(ids) >= config.max_len:  # 长则弃掉
        print u'pos 输出片段超过%d部分无法处理' % (config.max_len) 
        return ids[:config.max_len]
    ids.extend([0]*(config.max_len-len(ids))) # 短则补全
    ids = np.asarray(ids).reshape([-1, config.max_len])
    return ids

def show_result(tags):
    str = ''
    for (w, t) in tags:
        for i in range(len(t)):
            str += u'%s/%s '%(w[i], t[i])
    print "POS标记结果为: \n", str

def main():

    ckpt_path = '../ckpt/bi-lstm.ckpt-6'
    model = ModelLoader(ckpt_path)

    start = time.clock()
    with open('../data/pkl/dict_data.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
    
    sentence = u'词法 分析 终于 完成 了 。 这 都 叫 啥 事 啊 。'
    tagging = model.predict(model, sentence, word2id, id2tag)
    
    show_result(tagging)
    print start - time.clock() , " s"

if __name__ == '__main__':
    main()
