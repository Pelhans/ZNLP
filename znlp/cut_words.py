#!/usr/bin/env python
# coding=utf-8

import re
import time, sys,os
import glob
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path + r'/../lexical_analysis/cws_blstm/bin/')
import cws_blstm as cb

"""
Please use the command like "python cws_test.py 512 not_get_zy".
batch_size control the nums of sentences you process in one batch and not_get_zy/get_zy means not/generate zy.pkl(transition possibiy file).
"""


config = cb.config_ch()
#batch_size = 128
#wget_zy = 'not_get_zy'
#if len(sys.argv) == 1:
#    print "Using default config. batch_size = 128 and not_get_zy."
#elif len(sys.argv) == 3:
#    batch_size = sys.argv[1]
#    wget_zy = sys.argv[2]

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
    if len(ids) >= config.max_len:  # 长则弃掉
        print u'cws 输出片段超过%d部分无法处理' % (config.max_len) 
        return ids[:config.max_len]
    ids.extend([0]*(config.max_len-len(ids))) # 短则补全
    ids = np.asarray(ids).reshape([-1, config.max_len])
    return ids


def simple_cut(text, sen_words, model, zy, len_sen):
    """对一个片段text（标点符号把句子划分为多个片段）进行预测。"""
    if text:
        text_len = len(text)
        X_batch = np.squeeze(text, axis=(1,))
        fetches = [model.y_pred]
        feed_dict = {model.X_inputs:X_batch, config.lr:1.0, config.batch_size:len(text), config.keep_prob:1.0}
        eed_dict = {model.X_inputs:X_batch, config.lr:1.0, config.batch_size:len(text), config.keep_prob:1.0}
        _y_pred = model.session.run(fetches, feed_dict)
        _y_pred = np.squeeze(_y_pred, axis=(0,))
        y_pred = []
        for i in range(text_len):
            y_pred.append(_y_pred[i*config.max_len:i*config.max_len + len_sen[i]]) # remove padding
        words = []
        for i, tag_eachsen in enumerate(y_pred):
            nodes = [dict(zip(['s','b','m','e'], each[1:])) for each in tag_eachsen]
            tags = viterbi(nodes, zy)
            tmp = []
            for j in range(len_sen[i]):
                if tags[j] in ['s', 'n']:
                    words.append(sen_words[i][j])
                else:
                    tmp.extend(sen_words[i][j])
                    if tags[j] == 'e':
                        words.append(tmp)
                        tmp = []
        return words
    else:
        return []


def cut_word(sentence, word2id, model, zy, batch_size=128):
    """首先将一个sentence根据标点和英文符号/字符串划分成多个片段text，然后对每一个片段分词。"""
    not_cuts = re.compile(u'([0-9\da-zA-Z ]+)|[。，、？！.\.\?,!]')
    result = []
    sen_part_ids = []
    sen_part_words = []
    len_sen = []
    start = 0
    for seg_sign in not_cuts.finditer(sentence):
        sen_part_ids.append(text2ids(sentence[start:seg_sign.end()], word2id))
        sen_part_words.append(sentence[start:seg_sign.end()])
        len_sen.append(len(sentence[start:seg_sign.end()]))
        start = seg_sign.end()
    total_sen_num = len(sen_part_ids)
    print "total_sen_num: ", total_sen_num
    if total_sen_num > batch_size:
        for i in range(total_sen_num/batch_size):
            result.extend(simple_cut(sen_part_ids[i*batch_size:(i+1)*batch_size], sen_part_words[i*batch_size:(i+1)*batch_size], model, zy, len_sen[i*batch_size:(i+1)*batch_size]))
        result.extend(simple_cut(sen_part_ids[-total_sen_num%batch_size:], sen_part_words[-total_sen_num%batch_size:], model, zy, len_sen[-total_sen_num%batch_size:]))
    else:
        result.extend(simple_cut(sen_part_ids, sen_part_words, model, zy, len_sen))
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

def show_result(result):
    rss = ''
    for each in result:
        if isinstance(each, list):
            each = "".join(each)
        rss = rss + each + ' / '
    print rss

def main():

    sentence = u'人们思考问题往往不是从零开始的。就好像你现在阅读这篇文章一样，你对每个词的理解都会依赖于你前面看到的一些词，\
      而不是把你前面看的内容全部抛弃了，忘记了，再去理解这个单词。也就是说，人们的思维总是会有延续性的。'
    start = time.clock()
    mcws = zcws()
    result = mcws.zcws_cut(sentence)
    show_result(result)
    print time.clock() - start, "s"

if __name__ == '__main__':
    main()
