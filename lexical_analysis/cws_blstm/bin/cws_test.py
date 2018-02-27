#!/usr/bin/env python
# coding=utf-8

import re
import time, sys
import glob
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import cws_blstm as cb
from sklearn.model_selection import train_test_split

config = cb.config_ch()
class ModelLoader(object):
    def __init__(self,ckpt_path):
        self.session = tf.Session()
        self.ckpt_path = ckpt_path
        self.X_inputs =  tf.placeholder(tf.int32, [None, config.timestep_size], name='X_inputs')
        self.y_inputs = tf.placeholder(tf.int32, [None, config.timestep_size], name='y_input')

        with tf.variable_scope('cws_blstm'):
            self.cost, self.accuracy, self.correct_prediction, self.y_pred = cb.bi_lstm(self.X_inputs, self.y_inputs)
        if len(glob.glob(self.ckpt_path + '.data*')) > 0:
            print 'Loading model parameters from %s ' % ckpt_path
            all_vars = tf.global_variables()
            model_vars = [k for k in all_vars if k.name.startswith("cws_blstm")]
            tf.train.Saver(model_vars).restore(self.session, ckpt_path)
        else:
            print 'Model not found, creat with fresh parameters....'
            self.session.run(tf.global_variables_initializer())

#    def _init_cws_model(self, session, ckpt_path):
#        batch_size = 1
#        model = cb.bi_lstm(self.X_inputs, self.y_inputs, embedding)
#        if len(glob.glob(self.ckpt_path + 'bi-lstm-6*')) > 0:
#            print 'Loading model parameters from %s ' % ckpt_path
#            all_vars = tf.global_variables()
#            tf.train.Saver(all_vars).restore(session, ckpt_path)
#        else:
#            print 'Model not found, creat with fresh parameters....'
#            session.run(tf.global_variables_initializer())
#        return model

def viterbi(nodes, zy):
    """
    维特比译码：除了第一层以外，每一层有4个节点。
    计算当前层（第一层不需要计算）四个节点的最短路径：
       对于本层的每一个节点，计算出路径来自上一层的各个节点的新的路径长度（概率）。保留最大值（最短路径）。
       上一层每个节点的路径保存在 paths 中。计算本层的时候，先用paths_ 暂存，然后把本层的最大路径保存到 paths 中。
       paths 采用字典的形式保存（路径：路径长度）。
       一直计算到最后一层，得到四条路径，将长度最短（概率值最大的路径返回）
    """
    paths = {'b': nodes[0]['b'], 's':nodes[0]['s']} # 第一层，只有两个节点
    for layer in xrange(1, len(nodes)):  # 后面的每一层
        paths_ = paths.copy()  # 先保存上一的路径
        # node_now 为本层节点， node_last 为上层节点
        paths = {}  # 清空 path 
        for node_now in nodes[layer].keys():
            # 对于本层的每个节点，找出最短路径
            sub_paths = {} 
            # 上一层的每个节点到本层节点的连接
            for path_last in paths_.keys():
                if path_last[-1] + node_now in zy.keys(): # 若转移概率不为 0 
                    sub_paths[path_last + node_now] = paths_[path_last] + nodes[layer][node_now] + zy[path_last[-1] + node_now]
            # 最短路径,即概率最大的那个
            sr_subpaths = pd.Series(sub_paths)
            sr_subpaths = sr_subpaths.sort_values()  # 升序排序
            node_subpath = sr_subpaths.index[-1]  # 最短路径
            node_value = sr_subpaths[-1]   # 最短路径对应的值
            # 把 node_now 的最短路径添加到 paths 中
            paths[node_subpath] = node_value
    # 所有层求完后，找出最后一层中各个节点的路径最短的路径
    sr_paths = pd.Series(paths)
    sr_paths = sr_paths.sort_values()  # 按照升序排序
    return sr_paths.index[-1]  # 返回最短路径（概率值最大的路径）


def text2ids(text, word2id):
    """把字片段text转为 ids."""
    words = list(text)
    ids = list(word2id[word] if word in word2id.index else word2id["UNK"] for word in words)
    #ids = list(word2id[words])
    if len(ids) >= config.max_len:  # 长则弃掉
        print u'cws 输出片段超过%d部分无法处理' % (config.max_len) 
        return ids[:config.max_len]
    ids.extend([0]*(config.max_len-len(ids))) # 短则补全
    ids = np.asarray(ids).reshape([-1, config.max_len])
    return ids


def simple_cut(text, word2id, model, zy):
    """对一个片段text（标点符号把句子划分为多个片段）进行预测。"""
    if text:
        text_len = len(text)
        X_batch = text2ids(text, word2id)  # 这里每个 batch 是一个样本
        fetches = [model.y_pred]
        feed_dict = {model.X_inputs:X_batch, config.lr:1.0, config.batch_size:1, config.keep_prob:1.0}
        _y_pred = model.session.run(fetches, feed_dict)[0][:text_len]  # padding填充的部分直接丢弃
        nodes = [dict(zip(['s','b','m','e'], each[1:])) for each in _y_pred]
        tags = viterbi(nodes, zy)
        words = []
        for i in range(len(text)):
            if tags[i] in ['s', 'n']:
                words.append(text[i])
            else:
                words[-1] += text[i]
        return words
    else:
        return []


def cut_word(sentence, word2id, model, zy):
    """首先将一个sentence根据标点和英文符号/字符串划分成多个片段text，然后对每一个片段分词。"""
    not_cuts = re.compile(u'([0-9\da-zA-Z ]+)|[。，、？！.\.\?,!]')
    result = []
    start = 0
    for seg_sign in not_cuts.finditer(sentence):
        print sentence[start:seg_sign.start()]
        result.extend(simple_cut(sentence[start:seg_sign.start()], word2id, model, zy))
        result.append(sentence[seg_sign.start():seg_sign.end()])
        start = seg_sign.end()
    result.extend(simple_cut(sentence[start:], word2id, model, zy))
    return result

def get_zy(ltags):
    A = {
        'sb': 0,
        'ss': 0,
        'be': 0,
        'bm': 0,
        'me': 0,
        'mm': 0,
        'eb': 0,
        'es': 0
    }
    zy = dict()
    for label in ltags:
        for t in xrange(len(label) -1):
            key = label[t] + label[t+1]
            A[key] += 1.0

    zy['sb'] = A['sb'] / (A['sb'] + A['ss'])
    zy['ss'] = 1.0 - zy['sb']
    zy['be'] = A['be'] / (A['be'] + A['bm'])
    zy['bm'] = 1.0 - zy['be']
    zy['me'] = A['me'] / (A['me'] + A['mm'])
    zy['mm'] = 1.0 - zy['me']
    zy['eb'] = A['eb'] / (A['eb'] + A['es'])
    zy['es'] = 1.0 - zy['eb']
    keys = sorted(zy.keys())
    print 'the transition probability: '
    for key in keys:
        print key, zy[key]
    
    zy = {i:np.log(zy[i]) for i in zy.keys()}
    start_time = time.clock()
    with open ('../data/zy.pkl', 'wb') as outp:
        pickle.dump(zy, outp)
    end_time = time.clock()
    print start_time - end_time

def main():
    ckpt_path = '../ckpt/bi-lstm.ckpt-6'
    model = ModelLoader(ckpt_path)

    with open('../data/word2id.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        if sys.argv[1] == "getzy":
            ltags = pickle.load(inp)

    if sys.argv[1] == "getzy":
        get_zy(ltags)  #这行用来生成转移概率的pkl文件，一次生成后就可以注释掉了
    
    with open('../data/zy.pkl', 'rb') as inp:
        zy = pickle.load(inp)

#    sentence = u'人们思考问题往往不是从零开始的。就好像你现在阅读这篇文章一样，你对每个词的理解都会依赖于你前面看到的一些词，\
#      而不是把你前面看的内容全部抛弃了，忘记了，再去理解这个单词。也就是说，人们的思维总是会有延续性的。'
    #sentence = u'我爱吃北京烤鸭。'
    start = time.clock()
    sentence = u'他直言：“我没有参加台湾婚礼"'#'，所以这次觉得蛮开心。”'
    result = cut_word(sentence ,word2id ,model, zy)
    rss = ''
    for each in result:
        rss = rss + each + ' / '
    print rss
    print time.clock() - start, "s"

if __name__ == '__main__':
    main()
