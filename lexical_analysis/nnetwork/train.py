#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from sklearn.model_selection import train_test_split
from model.net_work import Model
from model.get_data import BatchGenerator
import argparse
from configparser import ConfigParser

parser = argparse.ArgumentParser()
parser.add_argument('--taskName', type=str, default='ner', 
                   help='the lexical task name, one of "ner" "pos" "cws"')
args = parser.parse_args()

cfg = ConfigParser()
cfg.read(u'conf/' + args.taskName + '_conf.ini')
print (cfg.get('file_path', 'train'))

def train():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        with open(cfg.get('file_path', 'train'), 'rb') as inp:
            X_train = pickle.load(inp)
            y_train = pickle.load(inp)
        with open(cfg.get('file_path', 'dev'), 'rb') as inp1:
            X_valid = pickle.load(inp1)
            y_valid = pickle.load(inp1)
        with open(cfg.get('file_path', 'test'), 'rb') as inp2:
            X_test = pickle.load(inp2)
            y_test = pickle.load(inp2)
        print('X_train.shape={}, y_train.shape={}; \n\
              X_valid.shape={}, y_valid.shape={};\n\
              X_test.shape={}, y_test.shape={}'.format(\
                                                       X_train.shape, y_train.shape,\
                                                       X_valid.shape, y_valid.shape, \
                                                       X_test.shape, y_test.shape))

        print('Creating the data generator ...')
        data_train = BatchGenerator(X_train, y_train, shuffle=True)
        data_valid = BatchGenerator(X_valid, y_valid, shuffle=False)
        data_test = BatchGenerator(X_test, y_test, shuffle=False)
        print ('Finished creating the data generator.')

        with tf.variable_scope('inputs'):
            X_inputs = tf.placeholder(tf.int32, [None, cfg.getint('net_work', 'timestep_size')], name='X_input')
            y_inputs = tf.placeholder(tf.int32, [None, cfg.getint('net_work', 'timestep_size')], name='y_input')   
        with tf.variable_scope(args.taskName + "_blstm", reuse=None):
            model = Model(cfg)
            cost, accuracy, correct_prediction, _ = model.bi_lstm(X_inputs, y_inputs)
    
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), cfg.getfloat('net_work', 'max_grad_norm'))
        optimizer = tf.train.AdamOptimizer(learning_rate=cfg.getfloat('net_work', 'lr'))
        train_op = optimizer.apply_gradients( zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        print('Finished creating the bi-lstm model.')
        sess.run(tf.global_variables_initializer())
        
        max_max_epoch = cfg.getint('net_work', 'max_max_epoch')
        max_epoch = cfg.getint('net_work', 'max_epoch')
        tr_batch_size = cfg.getint('net_work', 'batch_size')
        tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)
        display_batch = int(tr_batch_num / cfg.getint('net_work', 'display_num'))
        saver = tf.train.Saver(max_to_keep=1)
        for epoch in xrange(max_max_epoch):
            _lr = cfg.getfloat('net_work', 'lr')
            if epoch > max_epoch:
                _lr = _lr * ((cfg.getfloat('net_work', 'decay')) ** (epoch - max_epoch))
            print('EPOCH %d， lr=%g' % (epoch+1, _lr))
            start_time = time.time()
            train_cost = train_acc = valid_acc = valid_cost = 0.0
            train_acc, train_cost = model.run(data_train, sess, accuracy, cost, train_op,  X_inputs, y_inputs,  tr_batch_size, _lr, training=True)
            valid_acc, valid_cost = model.run(data_valid, sess,
                                                accuracy, cost, train_op, X_inputs, y_inputs, tr_batch_size, 0.0)
            print('\tEpoch 1: training acc=%g, cost=%g;  valid acc= %g, cost=%g speed=%g s/epoch'\
                          % (train_acc, train_cost, valid_acc, valid_cost, time.time()-start_time))
            if (epoch + 1) % 3 == 0:
                save_path = saver.save(sess, cfg.get('file_path', 'model'), global_step=(epoch+1))
                print('the save path is ', save_path)
        # testing
        print('**TEST RESULT:')
        test_acc, test_cost = model.run(data_test, sess, accuracy, cost, train_op, X_inputs, y_inputs, tr_batch_size, 0.0)
        print('**Test %d, acc=%g, cost=%g' % (data_test.y.shape[0], test_acc, test_cost))

if __name__ == "__main__":
    train()
