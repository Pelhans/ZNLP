#!/usr/bin/env python
# coding=utf-8

import os
import tensorflow as tf
from tensorflow.contrib import rnn
from configparser import ConfigParser

#cfg = ConfigParser()
#cfg.read(os.path.dirname(__file__) + '/../conf/' + 'ner' + '_conf.ini')

class Model(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.lr = tf.placeholder(tf.float32, [])
        self.batch_size = tf.placeholder(tf.int32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])

    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(self, shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    def lstm_cell(self):
        cell = rnn.LSTMCell(self.cfg.getint('net_work', 'hidden_size'), reuse=tf.get_variable_scope().reuse)
        return rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
             
    def bi_lstm(self, X_inputs, y_inputs):
        """build the bi-LSTMs network. Return the y_pred"""
        # X_inputs.shape = [batchsize, timestep_size]  ->  inputs.shape = [batchsize, timestep_size, embedding_size]
        embedding = tf.get_variable("ner_embedding", [self.cfg.getint('net_work', 'vocab_size'), self.cfg.getint('net_work', 'embedding_size')], dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, X_inputs)  
    
        # ** 1.构建前向后向多层 LSTM
        cell_fw = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.cfg.getint('net_work', 'layer_num'))], state_is_tuple=True)
        cell_bw = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.cfg.getint('net_work', 'layer_num'))], state_is_tuple=True)
    
        # ** 2.初始状态
        initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
        initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)  
    
        # ** 3. bi-lstm 计算（展开）
        with tf.variable_scope('bidirectional_rnn'):
            # Forward direction
            outputs_fw = list()
            state_fw = initial_state_fw
            with tf.variable_scope('fw'):
                for timestep in range(self.cfg.getint('net_work', 'timestep_size')):
                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    (output_fw, state_fw) = cell_fw(inputs[:, timestep, :], state_fw)
                    outputs_fw.append(output_fw)
        
            # backward direction
            outputs_bw = list()
            state_bw = initial_state_bw
            with tf.variable_scope('bw') as bw_scope:
                inputs = tf.reverse(inputs, [1])
                for timestep in range(self.cfg.getint('net_work', 'timestep_size')):
                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    (output_bw, state_bw) = cell_bw(inputs[:, timestep, :], state_bw)
                    outputs_bw.append(output_bw)
            outputs_bw = tf.reverse(outputs_bw, [0])
            output = tf.concat([outputs_fw, outputs_bw], 2)
            output = tf.transpose(output, perm=[1,0,2])
            output = tf.reshape(output, [-1, self.cfg.getint('net_work', 'hidden_size')*2])
            with tf.variable_scope('outputs'):
                softmax_w = self.weight_variable([self.cfg.getint('net_work', 'hidden_size') * 2, self.cfg.getint('net_work', 'class_num')]) 
                softmax_b = self.bias_variable([self.cfg.getint('net_work','class_num')]) 
                y_pred = tf.matmul(output, softmax_w) + softmax_b
    
            # adding extra statistics to monitor
            # y_inputs.shape = [batch_size, timestep_size]
            correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, 1), tf.int32), tf.reshape(y_inputs, [-1]))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(y_inputs, [-1]), logits = y_pred))
            return cost, accuracy, correct_prediction, y_pred
    
    def run(self, dataset, sess, accuracy, cost, train_op, X_inputs, y_inputs, _batch_size, training=False):
        """Run total dataset"""
        fetches = [accuracy, cost, train_op] if training else [accuracy, cost]
        keep_prob = self.cfg.getfloat('net_work', 'keep_prob') if training else 1.0
        _y = dataset.y
        data_size = _y.shape[0]
        batch_num = int(data_size / _batch_size)
        _acc = 0.0
        _cost = 0.0
    
        for i in xrange(batch_num):
            X_batch, y_batch = dataset.next_batch(_batch_size)
            feed_dict = {X_inputs:X_batch, y_inputs:y_batch, self.lr: self.cfg.getfloat('net_work', 'lr'), self.batch_size:_batch_size, self.keep_prob:keep_prob}
            result = sess.run(fetches, feed_dict)
            _acc += result[0]
            _cost += result[1]
        return _acc/batch_num, _cost/batch_num
