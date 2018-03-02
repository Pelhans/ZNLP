#!/usr/bin/env python
# coding=utf-8

import pickle
import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split

if not os.path.exists('../data/'):
    os.makedirs('../data/')

class BatchGenerator(object):
    """ Construct a Data generator. The input X, y should be ndarray or list like type.
    
    Example:
        Data_train = BatchGenerator(X=X_train_all, y=y_train_all, shuffle=False)
        Data_test = BatchGenerator(X=X_test_all, y=y_test_all, shuffle=False)
        X = Data_train.X
        y = Data_train.y
        or:
        X_batch, y_batch = Data_train.next_batch(batch_size)
     """ 
    
    def __init__(self, X, y, shuffle=False):
        if type(X) != np.ndarray:
            X = np.asarray(X)
        if type(y) != np.ndarray:
            y = np.asarray(y)
        self._X = X
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._number_examples = self._X.shape[0]
        self._shuffle = shuffle
        if self._shuffle:
            new_index = np.random.permutation(self._number_examples)
            self._X = self._X[new_index]
            self._y = self._y[new_index]
                
    @property
    def X(self):
        return self._X
    
    @property
    def y(self):
        return self._y
    
    @property
    def num_examples(self):
        return self._number_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size):
        #Return the next 'batch_size' examples from this data set.
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_examples:
            # finished epoch
            self._epochs_completed += 1
            # Shuffle the data 
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples)
                self._X = self._X[new_index]
                self._y = self._y[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples
        end = self._index_in_epoch
        return self._X[start:end], self._y[start:end]

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def lstm_cell():
    cell = rnn.LSTMCell(config_ch.hidden_size, reuse=tf.get_variable_scope().reuse)
    return rnn.DropoutWrapper(cell, output_keep_prob=config_ch.keep_prob)
         
def bi_lstm(X_inputs, y_inputs):
    """build the bi-LSTMs network. Return the y_pred"""
    # X_inputs.shape = [batchsize, timestep_size]  ->  inputs.shape = [batchsize, timestep_size, embedding_size]
    embedding = tf.get_variable("ner_embedding", [config_ch.vocab_size, config_ch.embedding_size], dtype=tf.float32)
    inputs = tf.nn.embedding_lookup(embedding, X_inputs)  

    # ** 1.构建前向后向多层 LSTM
    cell_fw = rnn.MultiRNNCell([lstm_cell() for _ in range(config_ch.layer_num)], state_is_tuple=True)
    cell_bw = rnn.MultiRNNCell([lstm_cell() for _ in range(config_ch.layer_num)], state_is_tuple=True)

    # ** 2.初始状态
    initial_state_fw = cell_fw.zero_state(config_ch.batch_size, tf.float32)
    initial_state_bw = cell_bw.zero_state(config_ch.batch_size, tf.float32)  

    # ** 3. bi-lstm 计算（展开）
    with tf.variable_scope('bidirectional_rnn'):
        # Forward direction
        outputs_fw = list()
        state_fw = initial_state_fw
        with tf.variable_scope('fw'):
            for timestep in range(config_ch.timestep_size):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (output_fw, state_fw) = cell_fw(inputs[:, timestep, :], state_fw)
                outputs_fw.append(output_fw)
    
        # backward direction
        outputs_bw = list()
        state_bw = initial_state_bw
        with tf.variable_scope('bw') as bw_scope:
            inputs = tf.reverse(inputs, [1])
            for timestep in range(config_ch.timestep_size):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (output_bw, state_bw) = cell_bw(inputs[:, timestep, :], state_bw)
                outputs_bw.append(output_bw)
        outputs_bw = tf.reverse(outputs_bw, [0])
        output = tf.concat([outputs_fw, outputs_bw], 2)
        output = tf.transpose(output, perm=[1,0,2])
        output = tf.reshape(output, [-1, config_ch.hidden_size*2])
        with tf.variable_scope('outputs'):
            softmax_w = weight_variable([config_ch.hidden_size * 2, config_ch.class_num]) 
            softmax_b = bias_variable([config_ch.class_num]) 
            y_pred = tf.matmul(output, softmax_w) + softmax_b

        # adding extra statistics to monitor
        # y_inputs.shape = [batch_size, timestep_size]
        correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, 1), tf.int32), tf.reshape(y_inputs, [-1]))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(y_inputs, [-1]), logits = y_pred))
        return cost, accuracy, correct_prediction, y_pred


def test_epoch(dataset, sess, accuracy, cost, X_inputs, y_inputs):
    """Testing or valid."""
    _batch_size = 64
    fetches = [accuracy, cost]
    _y = dataset.y
    data_size = _y.shape[0]
    batch_num = int(data_size / _batch_size)
    start_time = time.time()
    _costs = 0.0
    _accs = 0.0

    for i in xrange(batch_num):
        X_batch, y_batch = dataset.next_batch(_batch_size)
        feed_dict = {X_inputs:X_batch, y_inputs:y_batch, config_ch.lr:1e-4, config_ch.batch_size:_batch_size, config_ch.keep_prob:1.0}
        _acc, _cost = sess.run(fetches, feed_dict)
        _accs += _acc
        _costs += _cost    
    mean_acc= _accs / batch_num     
    mean_cost = _costs / batch_num
    return mean_acc, mean_cost


class config_ch():
    decay = 0.85
    max_epoch = 5
    max_max_epoch = 6
    timestep_size = max_len = 50
    vocab_size = 100000
    input_size = embedding_size = 128
    class_num = 5
    hidden_size = 256
    layer_num = 2
    max_grad_norm = 5.0
    lr = tf.placeholder(tf.float32, [])
    keep_prob = tf.placeholder(tf.float32, [])
    batch_size = tf.placeholder(tf.int32, [])
    model_save_path = '../ckpt/bi-lstm.ckpt'

def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
#    sess = tf.Session(config=config)
    with tf.Session(config=config) as sess:

        with open('../data/pkl/train_data.pkl', 'rb') as inp:
            X_train = pickle.load(inp)
            y_train = pickle.load(inp)
    
        with open('../data/pkl/dev_data.pkl', 'rb') as inp1:
            X_valid = pickle.load(inp1)
            y_valid = pickle.load(inp1)
    
        with open('../data/pkl/test_data.pkl', 'rb') as inp2:
            X_test = pickle.load(inp2)
            y_test = pickle.load(inp2)
    
        print 'X_train.shape={}, y_train.shape={}; \nX_valid.shape={}, y_valid.shape={};\nX_test.shape={}, y_test.shape={}'.format(
            X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape)
        print 'Creating the data generator ...'
        data_train = BatchGenerator(X_train, y_train, shuffle=True)
        data_valid = BatchGenerator(X_valid, y_valid, shuffle=False)
        data_test = BatchGenerator(X_test, y_test, shuffle=False)
        print 'Finished creating the data generator.'

        with tf.variable_scope('Inputs'):
            X_inputs = tf.placeholder(tf.int32, [None, config_ch.timestep_size], name='X_input')
            y_inputs = tf.placeholder(tf.int32, [None, config_ch.timestep_size], name='y_input')   
        with tf.variable_scope("ner_blstm", reuse=None):
            cost, accuracy, correct_prediction, _ = bi_lstm(X_inputs, y_inputs)
    
        # ***** 优化求解 *******
        tvars = tf.trainable_variables()  # 获取模型的所有参数
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config_ch.max_grad_norm)  # 获取损失函数对于每个参数的梯度
        optimizer = tf.train.AdamOptimizer(learning_rate=config_ch.lr)   # 优化器
    
        # 梯度下降计算
        train_op = optimizer.apply_gradients( zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        print 'Finished creating the bi-lstm model.'
    
        sess.run(tf.global_variables_initializer())
        tr_batch_size = 64
        max_max_epoch = config_ch.max_max_epoch
        display_num = 5  # 每个 epoch 显示是个结果
        tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)  # 每个 epoch 中包含的 batch 数
        display_batch = int(tr_batch_num / display_num)  # 每训练 display_batch 之后输出一次
        saver = tf.train.Saver(max_to_keep=1)  # 最多保存的模型数量
        for epoch in xrange(max_max_epoch):
            _lr = 1e-3
            if epoch > config_ch.max_epoch:
                _lr = _lr * ((config_ch.decay) ** (epoch - config_ch.max_epoch))
            print 'EPOCH %d， lr=%g' % (epoch+1, _lr)
            start_time = time.time()
            _costs = 0.0
            _accs = 0.0
            show_accs = 0.0
            show_costs = 0.0
            for batch in xrange(tr_batch_num): 
                fetches = [accuracy, cost, train_op]
                X_batch, y_batch = data_train.next_batch(tr_batch_size)
                feed_dict = {X_inputs:X_batch, y_inputs:y_batch, config_ch.lr:_lr, config_ch.batch_size:tr_batch_size, config_ch.keep_prob:0.5}
                _acc, _cost, _ = sess.run(fetches, feed_dict) # the cost is the mean cost of one batch
                _accs += _acc
                _costs += _cost
                show_accs += _acc
                show_costs += _cost
                if (batch + 1) % display_batch == 0:
                    valid_acc, valid_cost = test_epoch(data_valid, sess, accuracy, cost, X_inputs, y_inputs)
                    print '\ttraining acc=%g, cost=%g;  valid acc= %g, cost=%g ' % (show_accs / display_batch, show_costs / display_batch, valid_acc, valid_cost)
                    show_accs = 0.0
                    show_costs = 0.0
            mean_acc = _accs / tr_batch_num 
            mean_cost = _costs / tr_batch_num
            if (epoch + 1) % 3 == 0:  # 每 3 个 epoch 保存一次模型
                save_path = saver.save(sess, config_ch.model_save_path, global_step=(epoch+1))
                print 'the save path is ', save_path
            print '\ttraining %d, acc=%g, cost=%g ' % (data_train.y.shape[0], mean_acc, mean_cost)
            print 'Epoch training %d, acc=%g, cost=%g, speed=%g s/epoch' % (data_train.y.shape[0], mean_acc, mean_cost, time.time()-start_time)        
        # testing
        print '**TEST RESULT:'
        test_acc, test_cost = test_epoch(data_test, sess, accuracy, cost, X_inputs, y_inputs)
        print '**Test %d, acc=%g, cost=%g' % (data_test.y.shape[0], test_acc, test_cost)

if __name__ == "__main__":
    main()
