#!/usr/bin/env python
# coding=utf-8

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain
from functools import partial
import argparse
from configparser import ConfigParser

parser = argparse.ArgumentParser()
parser.add_argument('--taskName', type=str, default='ner', 
                   help='the lexical task name, one of "ner" "pos" "cws"')
parser.add_argument('--max_len', type=int, default=50,
                   help='max length of one sentence')
args = parser.parse_args()

cfg = ConfigParser()
cfg.read(u'../conf/' + args.taskName + '_conf.ini')

def get_dict(filename):
    all_words = []
    all_tags = []
    with open(os.path.join(filename) , "r") as inp:
        lines = inp.readlines()
        for line in lines:
            words = []
            tags = []
            words_tags = line.strip().split(" ")
            for word_tag in words_tags:
                word_tag = word_tag.split("/")
                if len(word_tag) != 2:
                    continue
                words.append(word_tag[0].decode('utf-8'))
                tags.append(word_tag[1].decode('utf-8'))
            all_words.append(words)
            all_tags.append(tags)
    return all_words, all_tags

def build_dict(df_data):
    # 构建词 与对应索引
    all_words = list(chain(*df_data['words'].values))
    all_words.append(u'UNK')
    sr_all_words = pd.Series(all_words)
    sr_all_words =sr_all_words.value_counts()
    set_words = sr_all_words.index
    set_ids = range(1, len(set_words) + 1)
    # 构建类别标签与索引
    tags = read_list( cfg.get('get_pkl', 'tags') )
    print "tags: ", type(tags), tags
#    tags = ['nz', 'nt', 'ns', 'nr', 'nan']
    tag_ids = range(len(tags))

    # 构建 词->id、id->词、标签->id、id->标签的映射
    word2id = pd.Series(set_ids, index=set_words)
    id2word = pd.Series(set_words, index = set_ids)
    tag2id = pd.Series(tag_ids, index = tags)
    id2tag = pd.Series(tags, index = tag_ids)

    vocab_size = len(set_words)
    print 'vocab_size={}'.format(vocab_size)

    return word2id, id2word, tag2id, id2tag

def padding(input, map_dict):
    total_tag = list(map_dict.index)
    input = [i if i in total_tag else u'UNK' for i in input]
    ids = list(map_dict[input])
    if len(ids) >= args.max_len:
        return ids[:args.max_len]
    ids.extend([0]*(args.max_len - len(ids)))
    return ids

def text_to_ids(df_data, word2id, tag2id):
    df_data['X'] = df_data['words'].apply(padding, args=(word2id, ))
    df_data['y'] = df_data['tags'].apply(padding, args=(tag2id, ))
    X = np.asarray(list(df_data['X'].values))
    y = np.asarray(list(df_data['y'].values))

    return X, y

def read_list(list_str):
    return [i for i in list_str.split("'") if i not in ["[", "]", ",", ", "]]

def save_data(data_type, X, y):
    with open(cfg.get('get_pkl', 'pkl_path') + data_type + "_data.pkl" , "wb") as outp:
        pickle.dump(X, outp)
        pickle.dump(y, outp)

def save_dict(word2id, id2word, tag2id, id2tag):
    with open(cfg.get('get_pkl', 'pkl_path') + 'dict.pkl', 'wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)

if __name__ == '__main__':
    print "Generating", args.taskName, " file"
    word2id = id2word = tag2id = id2tag = pd.Series()
    # "train" should be the first one
    for filename in ["train", "dev", "test"]:
        words, tags = get_dict( cfg.get("get_pkl", "txt_path") + filename + ".txt")
        df_data = pd.DataFrame({'words': words, "tags": tags},
                               index = range(len(words)))
        if filename == "train":
            word2id, id2word, tag2id, id2tag = build_dict(df_data)
            save_dict(word2id, id2word, tag2id, id2tag)
        X, y = text_to_ids(df_data, word2id, tag2id)
        save_data(filename, X, y)
